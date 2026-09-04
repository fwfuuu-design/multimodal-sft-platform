# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified in 2026 for the Multimodal Fine-tuning Platform MVP.

"""Authentication, project RBAC, encrypted credentials, quotas, and audit.

The token verifier is injectable so production can use an OIDC verifier while
contract tests remain offline. Credential plaintext is accepted only by the
server-side provisioning method and is never returned by an API.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from cryptography.fernet import Fernet, InvalidToken

from .jobs import RandomIdGenerator, SystemClock


ROLES = {"admin", "trainer", "viewer", "agent"}
VISIBILITIES = {"private", "project", "company"}
ACTIVE_JOB_STATES = {"DRAFT", "VALIDATING", "QUEUED", "STARTING", "RUNNING", "STOPPING"}
DEFAULT_QUOTAS = {
    "max_total_jobs": 100,
    "max_concurrent_jobs": 2,
    "max_upload_bytes": 5 * 1024**3,
    "max_log_bytes_per_minute": 1024 * 1024,
}


class SecurityError(ValueError):
    def __init__(self, code: str, field: str, message: str, status_code: int = 403):
        self.code = code
        self.field = field
        self.status_code = status_code
        super().__init__(f"{code}: {message} (field={field!r})")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": str(self)}


@dataclass(frozen=True)
class Identity:
    subject: str
    display_name: str


@dataclass(frozen=True)
class Principal:
    user_id: str
    display_name: str
    role: str
    project_id: str | None


class TokenVerifier(Protocol):
    def verify(self, token: str) -> Identity: ...


class FakeOIDCVerifier:
    """Offline SSO/OIDC claim verifier for contract tests."""

    def __init__(self, identities: Mapping[str, Identity]):
        self.identities = dict(identities)

    def verify(self, token: str) -> Identity:
        try:
            return self.identities[token]
        except KeyError as error:
            raise SecurityError("AUTH_TOKEN_INVALID", "authorization", "Bearer token is invalid.", 401) from error


@dataclass(frozen=True)
class RequestDecision:
    principal: Principal
    project_id: str | None
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    sensitive: bool = False


class SecurityService:
    def __init__(
        self,
        database_path: str | Path,
        *,
        token_verifier: TokenVerifier,
        encryption_key: bytes,
        training_service: Any | None = None,
        clock: Any | None = None,
        id_generator: Callable[[str], str] | None = None,
        audit_retention_days: int = 180,
    ):
        self.database_path = str(Path(database_path).expanduser().resolve())
        self.token_verifier = token_verifier
        try:
            self.cipher = Fernet(encryption_key)
        except (TypeError, ValueError) as error:
            raise ValueError("A valid Fernet master key is required.") from error
        self.training_service = training_service
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or RandomIdGenerator()
        self.audit_retention_days = audit_retention_days
        self._initialize()

    @staticmethod
    def generate_encryption_key() -> bytes:
        return Fernet.generate_key()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS security_projects (
                    project_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS security_memberships (
                    project_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL,
                    display_name TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY (project_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS security_objects (
                    object_type TEXT NOT NULL, object_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL, visibility TEXT NOT NULL, created_at REAL NOT NULL,
                    deleted_at REAL, deletion_reason TEXT,
                    PRIMARY KEY (object_type, object_id)
                );
                CREATE TABLE IF NOT EXISTS security_credentials (
                    credential_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL, ciphertext BLOB NOT NULL, created_at REAL NOT NULL, revoked_at REAL
                );
                CREATE TABLE IF NOT EXISTS security_quotas (
                    project_id TEXT NOT NULL, user_id TEXT NOT NULL, quota_json TEXT NOT NULL,
                    PRIMARY KEY (project_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS security_rate_usage (
                    project_id TEXT NOT NULL, user_id TEXT NOT NULL, kind TEXT NOT NULL,
                    window_start INTEGER NOT NULL, used_bytes INTEGER NOT NULL,
                    PRIMARY KEY (project_id, user_id, kind, window_start)
                );
                CREATE TABLE IF NOT EXISTS security_audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at REAL NOT NULL,
                    user_id TEXT, role TEXT, project_id TEXT, action TEXT NOT NULL,
                    resource_type TEXT, resource_id TEXT, outcome TEXT NOT NULL,
                    status_code INTEGER NOT NULL, detail_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS security_audit_time_idx ON security_audit(occurred_at);
                CREATE INDEX IF NOT EXISTS security_object_project_idx
                    ON security_objects(project_id, object_type, deleted_at);
                """
            )

    def bootstrap_project(
        self,
        name: str,
        admin_user_id: str,
        admin_display_name: str | None = None,
    ) -> dict[str, Any]:
        """Create the first project exactly once without exposing an anonymous HTTP route."""
        if not name.strip():
            raise SecurityError("PROJECT_NAME_REQUIRED", "name", "Project name is required.", 422)
        if not admin_user_id.strip():
            raise SecurityError("ADMIN_SUBJECT_REQUIRED", "admin_user_id", "Initial administrator subject is required.", 422)
        project_id = self.id_generator("project")
        now = self.clock.now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM security_projects LIMIT 1").fetchone() is not None:
                raise SecurityError(
                    "BOOTSTRAP_ALREADY_COMPLETED",
                    "database",
                    "Initial project bootstrap is only allowed on an empty project database.",
                    409,
                )
            connection.execute("INSERT INTO security_projects VALUES (?, ?, ?)", (project_id, name.strip(), now))
            connection.execute(
                "INSERT INTO security_memberships VALUES (?, ?, 'admin', ?, ?)",
                (project_id, admin_user_id.strip(), admin_display_name or admin_user_id.strip(), now),
            )
        principal = Principal(
            admin_user_id.strip(),
            admin_display_name or admin_user_id.strip(),
            "admin",
            project_id,
        )
        self.audit(
            principal,
            action="project.bootstrap",
            resource_type="project",
            resource_id=project_id,
            outcome="SUCCESS",
            status_code=201,
        )
        return {"id": project_id, "name": name.strip(), "admin_user_id": admin_user_id.strip(), "created": True}

    def create_project(self, name: str, admin_user_id: str, admin_display_name: str | None = None) -> dict[str, Any]:
        if not name.strip():
            raise SecurityError("PROJECT_NAME_REQUIRED", "name", "Project name is required.", 422)
        project_id = self.id_generator("project")
        now = self.clock.now()
        with self._connect() as connection:
            connection.execute("INSERT INTO security_projects VALUES (?, ?, ?)", (project_id, name.strip(), now))
            connection.execute(
                "INSERT INTO security_memberships VALUES (?, ?, 'admin', ?, ?)",
                (project_id, admin_user_id, admin_display_name or admin_user_id, now),
            )
        return {"id": project_id, "name": name.strip()}

    def add_member(self, project_id: str, user_id: str, display_name: str, role: str) -> dict[str, str]:
        if role not in ROLES:
            raise SecurityError("ROLE_INVALID", "role", "Role must be admin, trainer, viewer, or agent.", 422)
        self._ensure_project(project_id)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO security_memberships VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(project_id, user_id) DO UPDATE SET role=excluded.role, display_name=excluded.display_name",
                (project_id, user_id, role, display_name, self.clock.now()),
            )
        return {"project_id": project_id, "user_id": user_id, "display_name": display_name, "role": role}

    def authenticate(self, authorization: str | None, project_id: str | None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise SecurityError("AUTH_REQUIRED", "authorization", "OIDC Bearer token is required.", 401)
        identity = self.token_verifier.verify(authorization[7:])
        if project_id is None:
            memberships = self.memberships(identity.subject)
            if len(memberships) == 1:
                project_id = memberships[0]["project_id"]
            elif not memberships:
                raise SecurityError("PROJECT_MEMBERSHIP_REQUIRED", "project_id", "User has no project membership.")
            else:
                raise SecurityError("PROJECT_REQUIRED", "project_id", "X-Project-ID is required.", 422)
        with self._connect() as connection:
            membership = connection.execute(
                "SELECT role FROM security_memberships WHERE project_id = ? AND user_id = ?",
                (project_id, identity.subject),
            ).fetchone()
        if membership is None:
            raise SecurityError("PROJECT_ACCESS_DENIED", "project_id", "User is not a project member.")
        return Principal(identity.subject, identity.display_name, membership["role"], project_id)

    def memberships(self, user_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT project_id, role, display_name FROM security_memberships WHERE user_id = ? ORDER BY project_id",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def bind_object(
        self,
        principal: Principal,
        object_type: str,
        object_id: str,
        *,
        visibility: str = "project",
    ) -> None:
        if visibility not in VISIBILITIES:
            raise SecurityError("VISIBILITY_INVALID", "visibility", "Visibility is invalid.", 422)
        if not principal.project_id:
            raise SecurityError("PROJECT_REQUIRED", "project_id", "Project context is required.", 422)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO security_objects VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                (object_type, object_id, principal.project_id, principal.user_id, visibility, self.clock.now()),
            )

    def object_record(self, object_type: str, object_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM security_objects WHERE object_type = ? AND object_id = ?",
                (object_type, object_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def can_access(self, principal: Principal, object_type: str, object_id: str, *, write: bool = False) -> bool:
        record = self.object_record(object_type, object_id)
        if record is None or record["deleted_at"] is not None:
            return False
        if principal.role == "admin" and record["project_id"] == principal.project_id:
            return True
        if record["visibility"] == "company" and not write:
            return True
        if record["project_id"] != principal.project_id:
            return False
        if write:
            return principal.role == "trainer" and record["owner_id"] == principal.user_id
        return record["visibility"] == "project" or record["owner_id"] == principal.user_id

    def require_access(
        self, principal: Principal, object_type: str, object_id: str, *, write: bool = False
    ) -> dict[str, Any]:
        if not self.can_access(principal, object_type, object_id, write=write):
            raise SecurityError("RESOURCE_ACCESS_DENIED", object_type, "Resource is not visible or writable.", 404)
        record = self.object_record(object_type, object_id)
        assert record is not None
        return record

    def filter_items(self, principal: Principal, object_type: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in items if item.get("id") and self.can_access(principal, object_type, item["id"])]

    def set_visibility(
        self, principal: Principal, object_type: str, object_id: str, visibility: str
    ) -> dict[str, Any]:
        if visibility not in VISIBILITIES:
            raise SecurityError("VISIBILITY_INVALID", "visibility", "Visibility is invalid.", 422)
        record = self.require_access(principal, object_type, object_id, write=principal.role != "admin")
        with self._connect() as connection:
            connection.execute(
                "UPDATE security_objects SET visibility = ? WHERE object_type = ? AND object_id = ?",
                (visibility, object_type, object_id),
            )
        return {**record, "visibility": visibility}

    def provision_credential(
        self, principal: Principal, *, kind: str, plaintext: str
    ) -> dict[str, str]:
        if principal.role != "admin":
            raise SecurityError("ADMIN_REQUIRED", "role", "Only administrators provision credentials.")
        if not plaintext:
            raise SecurityError("CREDENTIAL_SECRET_REQUIRED", "plaintext", "Credential secret is required.", 422)
        credential_id = self.id_generator("credential")
        ciphertext = self.cipher.encrypt(plaintext.encode())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO security_credentials VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (credential_id, principal.project_id, principal.user_id, kind, ciphertext, self.clock.now()),
            )
        self.audit(
            principal,
            action="credential.provision",
            resource_type="credential",
            resource_id=credential_id,
            outcome="SUCCESS",
            status_code=201,
            detail={"kind": kind},
        )
        return {"id": credential_id, "kind": kind, "project_id": str(principal.project_id)}

    def resolve_credential(self, credential_id: str, project_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT ciphertext FROM security_credentials WHERE credential_id = ? AND project_id = ? "
                "AND revoked_at IS NULL",
                (credential_id, project_id),
            ).fetchone()
        if row is None:
            raise SecurityError("CREDENTIAL_NOT_FOUND", "credential_id", "Credential does not exist.", 404)
        try:
            return self.cipher.decrypt(row["ciphertext"]).decode()
        except InvalidToken as error:
            raise SecurityError("CREDENTIAL_DECRYPTION_FAILED", "credential_id", "Credential cannot be decrypted.", 500) from error

    def list_credentials(self, principal: Principal) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT credential_id, kind, project_id, owner_id, created_at, revoked_at "
                "FROM security_credentials WHERE project_id = ? ORDER BY created_at",
                (principal.project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def credential_exists(self, principal: Principal, credential_id: str) -> bool:
        return any(item["credential_id"] == credential_id and item["revoked_at"] is None for item in self.list_credentials(principal))

    def set_quota(self, project_id: str, user_id: str, values: Mapping[str, int]) -> dict[str, int]:
        unknown = set(values).difference(DEFAULT_QUOTAS)
        if unknown or any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise SecurityError("QUOTA_INVALID", "quota", "Quota names or values are invalid.", 422)
        quota = {**DEFAULT_QUOTAS, **dict(values)}
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO security_quotas VALUES (?, ?, ?) ON CONFLICT(project_id, user_id) "
                "DO UPDATE SET quota_json=excluded.quota_json",
                (project_id, user_id, json.dumps(quota, sort_keys=True)),
            )
        return quota

    def get_quota(self, principal: Principal) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT quota_json FROM security_quotas WHERE project_id = ? AND user_id = ?",
                (principal.project_id, principal.user_id),
            ).fetchone()
        return dict(DEFAULT_QUOTAS) if row is None else json.loads(row["quota_json"])

    def check_job_quota(self, principal: Principal) -> None:
        quota = self.get_quota(principal)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT object_id FROM security_objects WHERE object_type = 'training_job' AND project_id = ? "
                "AND owner_id = ? AND deleted_at IS NULL",
                (principal.project_id, principal.user_id),
            ).fetchall()
        if len(rows) >= quota["max_total_jobs"]:
            raise SecurityError("JOB_TOTAL_QUOTA_EXCEEDED", "quota", "User job quota is exhausted.", 429)
        if self.training_service is not None:
            concurrent = 0
            for row in rows:
                try:
                    if self.training_service.get_job(row["object_id"])["state"] in ACTIVE_JOB_STATES:
                        concurrent += 1
                except ValueError:
                    continue
            if concurrent >= quota["max_concurrent_jobs"]:
                raise SecurityError("JOB_CONCURRENCY_QUOTA_EXCEEDED", "quota", "Concurrent job quota is exhausted.", 429)

    def check_upload_quota(self, principal: Principal, request_bytes: int) -> None:
        if request_bytes > self.get_quota(principal)["max_upload_bytes"]:
            raise SecurityError("UPLOAD_QUOTA_EXCEEDED", "file", "Upload exceeds the user quota.", 413)

    def consume_log_quota(self, principal: Principal, request_bytes: int) -> None:
        quota = self.get_quota(principal)["max_log_bytes_per_minute"]
        window_start = int(self.clock.now() // 60) * 60
        with self._connect() as connection:
            row = connection.execute(
                "SELECT used_bytes FROM security_rate_usage WHERE project_id = ? AND user_id = ? "
                "AND kind = 'logs' AND window_start = ?",
                (principal.project_id, principal.user_id, window_start),
            ).fetchone()
            used = 0 if row is None else row["used_bytes"]
            if used + request_bytes > quota:
                raise SecurityError("LOG_RATE_QUOTA_EXCEEDED", "logs", "Log byte rate quota is exhausted.", 429)
            connection.execute(
                "INSERT INTO security_rate_usage VALUES (?, ?, 'logs', ?, ?) "
                "ON CONFLICT(project_id, user_id, kind, window_start) DO UPDATE SET used_bytes=excluded.used_bytes",
                (principal.project_id, principal.user_id, window_start, used + request_bytes),
            )

    def soft_delete(self, principal: Principal, object_type: str, object_id: str, reason: str) -> dict[str, Any]:
        if principal.role != "admin":
            raise SecurityError("ADMIN_REQUIRED", "role", "Only administrators delete managed data.")
        self.require_access(principal, object_type, object_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE security_objects SET deleted_at = ?, deletion_reason = ? WHERE object_type = ? AND object_id = ?",
                (self.clock.now(), reason, object_type, object_id),
            )
        return {"object_type": object_type, "object_id": object_id, "deleted": True, "physical_delete": False}

    def apply_retention(self, principal: Principal, *, older_than_days: int) -> dict[str, Any]:
        if principal.role != "admin":
            raise SecurityError("ADMIN_REQUIRED", "role", "Only administrators apply retention.")
        if older_than_days < 1:
            raise SecurityError("RETENTION_INVALID", "older_than_days", "Retention must be at least one day.", 422)
        threshold = self.clock.now() - older_than_days * 86400
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT object_type, object_id FROM security_objects WHERE project_id = ? AND deleted_at IS NULL "
                "AND created_at < ? AND object_type != 'node'",
                (principal.project_id, threshold),
            ).fetchall()
            connection.executemany(
                "UPDATE security_objects SET deleted_at = ?, deletion_reason = 'retention' "
                "WHERE object_type = ? AND object_id = ?",
                [(self.clock.now(), row["object_type"], row["object_id"]) for row in rows],
            )
        return {
            "soft_deleted": [dict(row) for row in rows],
            "physical_delete": False,
            "message": "Metadata is hidden; physical deletion requires a separately confirmed administrator operation.",
        }

    def audit(
        self,
        principal: Principal | None,
        *,
        action: str,
        resource_type: str | None,
        resource_id: str | None,
        outcome: str,
        status_code: int,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        safe_detail = {key: value for key, value in (detail or {}).items() if key not in {"token", "secret", "password"}}
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO security_audit(occurred_at, user_id, role, project_id, action, resource_type, "
                "resource_id, outcome, status_code, detail_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.clock.now(),
                    principal.user_id if principal else None,
                    principal.role if principal else None,
                    principal.project_id if principal else None,
                    action,
                    resource_type,
                    resource_id,
                    outcome,
                    status_code,
                    json.dumps(safe_detail, sort_keys=True),
                ),
            )

    def list_audit(self, principal: Principal, *, limit: int = 1000) -> list[dict[str, Any]]:
        if principal.role != "admin":
            raise SecurityError("ADMIN_REQUIRED", "role", "Only administrators view audit events.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM security_audit WHERE project_id = ? OR project_id IS NULL "
                "ORDER BY sequence DESC LIMIT ?",
                (principal.project_id, limit),
            ).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail_json"])} for row in rows]

    def prune_audit(self, principal: Principal) -> int:
        if principal.role != "admin":
            raise SecurityError("ADMIN_REQUIRED", "role", "Only administrators prune audit events.")
        threshold = self.clock.now() - self.audit_retention_days * 86400
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM security_audit WHERE project_id = ? AND occurred_at < ?",
                (principal.project_id, threshold),
            )
        return cursor.rowcount

    def principal_dict(self, principal: Principal) -> dict[str, Any]:
        return {**asdict(principal), "memberships": self.memberships(principal.user_id)}

    def _ensure_project(self, project_id: str) -> None:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM security_projects WHERE project_id = ?", (project_id,)
            ).fetchone() is None:
                raise SecurityError("PROJECT_NOT_FOUND", "project_id", "Project does not exist.", 404)
