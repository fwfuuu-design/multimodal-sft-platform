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

"""Database version and non-destructive backup/recovery planning gates."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DATABASE_SCHEMA_VERSION = 1
API_CONTRACT_VERSION = "1.0.0"
LOCKED_CODE_VERSION = "4451765a6b04ff08a6c5650f5953513608ae9e64"
RECOVERY_POLICY = {
    "rpo_hours": 24,
    "rto_hours": 4,
    "daily_backup_retention_days": 30,
    "monthly_backup_retention_months": 6,
    "recovery_exercise_interval_months": 3,
}
_OPERATION_ENVIRONMENT = (
    "MVP_DATABASE_PATH",
    "MVP_STATE_ROOT",
    "MVP_DATA_ROOTS",
    "MVP_MEDIA_ROOTS",
    "MVP_RESULTS_ROOT",
)


class OperationsError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def prepare_database_schema(database_path: str | Path) -> int:
    """Initialize or validate the control-plane schema without destructive migration."""
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise OperationsError("DATABASE_PATH_INVALID", "Database path must point to a file.")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
            "version INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        row = connection.execute("SELECT version FROM schema_version WHERE singleton = 1").fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO schema_version(singleton, version, updated_at) VALUES (1, ?, ?)",
                (DATABASE_SCHEMA_VERSION, _utc_now()),
            )
            return DATABASE_SCHEMA_VERSION
        version = int(row[0])
    if version > DATABASE_SCHEMA_VERSION:
        raise OperationsError(
            "DATABASE_SCHEMA_TOO_NEW",
            f"Database schema {version} is newer than supported schema {DATABASE_SCHEMA_VERSION}.",
        )
    if version < DATABASE_SCHEMA_VERSION:
        raise OperationsError(
            "DATABASE_MIGRATION_REQUIRED",
            "Automatic migration is disabled; create and approve a backup-backed migration plan.",
        )
    return version


def build_backup_plan(
    environment: Mapping[str, str] | None = None,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Describe a consistent backup set without copying or changing any file."""
    env = os.environ if environment is None else environment
    missing = [name for name in _OPERATION_ENVIRONMENT if not env.get(name)]
    if missing:
        raise OperationsError("BACKUP_CONFIGURATION_MISSING", f"Missing configuration: {', '.join(missing)}")

    database_path = Path(env["MVP_DATABASE_PATH"]).expanduser().resolve()
    if not database_path.is_file():
        raise OperationsError("BACKUP_DATABASE_NOT_FOUND", f"Database does not exist: {database_path}")
    schema_version = _read_schema_version(database_path)
    if schema_version != DATABASE_SCHEMA_VERSION:
        raise OperationsError("BACKUP_SCHEMA_UNSUPPORTED", f"Unsupported database schema: {schema_version}")

    sources: list[dict[str, Any]] = [_source("database", database_path)]
    sources.append(_source("state", _existing_directory(env["MVP_STATE_ROOT"], "MVP_STATE_ROOT")))
    sources.extend(_path_list_sources("data", env["MVP_DATA_ROOTS"], "MVP_DATA_ROOTS"))
    sources.extend(_path_list_sources("media", env["MVP_MEDIA_ROOTS"], "MVP_MEDIA_ROOTS"))
    sources.append(_source("results", _existing_directory(env["MVP_RESULTS_ROOT"], "MVP_RESULTS_ROOT")))
    sources.extend(_node_output_sources(database_path))
    sources = _deduplicate_sources(sources)

    return {
        "kind": "mvp-backup-plan",
        "dry_run": True,
        "physical_changes": False,
        "generated_at": generated_at or _utc_now(),
        "code_version": LOCKED_CODE_VERSION,
        "api_contract_version": API_CONTRACT_VERSION,
        "database_schema_version": schema_version,
        "recovery_policy": dict(RECOVERY_POLICY),
        "consistency_requirements": [
            "stop accepting new jobs",
            "quiesce or stop active writers",
            "use the SQLite online backup API or an approved write pause",
            "capture all sources in one identified backup batch",
            "record external key versions without exporting plaintext keys",
        ],
        "sources": sources,
    }


def build_restore_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a backup manifest and return steps; never restore or overwrite files."""
    if manifest.get("kind") != "mvp-backup-plan" or manifest.get("dry_run") is not True:
        raise OperationsError("RESTORE_MANIFEST_INVALID", "Manifest is not an MVP dry-run backup plan.")
    if manifest.get("code_version") != LOCKED_CODE_VERSION:
        raise OperationsError("RESTORE_CODE_VERSION_MISMATCH", "Backup code version does not match the locked baseline.")
    if manifest.get("api_contract_version") != API_CONTRACT_VERSION:
        raise OperationsError("RESTORE_API_VERSION_MISMATCH", "Backup API contract version is incompatible.")
    if manifest.get("database_schema_version") != DATABASE_SCHEMA_VERSION:
        raise OperationsError("RESTORE_SCHEMA_VERSION_MISMATCH", "Backup database schema is incompatible.")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources or not any(item.get("kind") == "database" for item in sources):
        raise OperationsError("RESTORE_SOURCES_INVALID", "Backup sources must include the database.")

    return {
        "kind": "mvp-restore-plan",
        "dry_run": True,
        "physical_changes": False,
        "source_generated_at": manifest.get("generated_at"),
        "code_version": LOCKED_CODE_VERSION,
        "api_contract_version": API_CONTRACT_VERSION,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "recovery_policy": dict(RECOVERY_POLICY),
        "steps": [
            "confirm the target environment and overwrite scope",
            "stop API and Worker scheduling",
            "verify backup batch identifiers and file digests",
            "restore database and directories as one approved batch",
            "inject the matching external key version",
            "start in an isolated network",
            "reconcile attempts, leases, events, logs, and artifacts with an isolated Worker",
            "open traffic only after duplicate execution checks pass",
        ],
    }


def load_restore_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationsError("RESTORE_MANIFEST_INVALID", "Restore manifest is missing or invalid JSON.") from error
    if not isinstance(payload, dict):
        raise OperationsError("RESTORE_MANIFEST_INVALID", "Restore manifest must be a JSON object.")
    return payload


def _read_schema_version(database_path: Path) -> int:
    try:
        with sqlite3.connect(database_path) as connection:
            row = connection.execute("SELECT version FROM schema_version WHERE singleton = 1").fetchone()
    except sqlite3.Error as error:
        raise OperationsError("DATABASE_SCHEMA_MISSING", "Database has no initialized schema version.") from error
    if row is None:
        raise OperationsError("DATABASE_SCHEMA_MISSING", "Database has no initialized schema version.")
    return int(row[0])


def _existing_directory(value: str, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise OperationsError("BACKUP_SOURCE_NOT_FOUND", f"{field} is not an existing directory: {path}")
    return path


def _path_list_sources(kind: str, value: str, field: str) -> list[dict[str, Any]]:
    paths = [_existing_directory(item, field) for item in value.split(os.pathsep) if item]
    if not paths:
        raise OperationsError("BACKUP_CONFIGURATION_MISSING", f"{field} must contain at least one directory.")
    return [_source(kind, path) for path in paths]


def _node_output_sources(database_path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(database_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'nodes'"
            ).fetchone()
            rows = [] if exists is None else connection.execute("SELECT output_root FROM nodes").fetchall()
    except sqlite3.Error as error:
        raise OperationsError("BACKUP_NODE_QUERY_FAILED", "Unable to inspect Worker output roots.") from error
    return [_source("worker_output", _existing_directory(row[0], "node.output_root")) for row in rows]


def _source(kind: str, path: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "type": "file" if path.is_file() else "directory",
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _deduplicate_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for source in sources:
        key = (source["kind"], source["path"])
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result


def run_backup_plan_cli(arguments: list[str]) -> None:
    """Print a backup plan; the command has no write or copy mode."""
    if arguments != ["--dry-run"]:
        raise OperationsError("BACKUP_DRY_RUN_REQUIRED", "Usage: sft-train backup --dry-run")
    print(json.dumps(build_backup_plan(), ensure_ascii=False, indent=2, sort_keys=True))


def run_restore_plan_cli(arguments: list[str]) -> None:
    """Print a restore plan; the command never overwrites target files."""
    if len(arguments) != 2 or arguments[0] != "--dry-run":
        raise OperationsError(
            "RESTORE_DRY_RUN_REQUIRED",
            "Usage: sft-train restore --dry-run <backup-plan.json>",
        )
    manifest = load_restore_manifest(arguments[1])
    print(json.dumps(build_restore_plan(manifest), ensure_ascii=False, indent=2, sort_keys=True))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
