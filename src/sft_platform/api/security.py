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

"""Authenticated project gateway and administration API for the MVP."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..mvp.security import Principal, RequestDecision, SecurityError, SecurityService


_RESOURCE_PATHS = (
    (re.compile(r"^/api/v1/models/(?P<id>model-[^/]+)$"), "model"),
    (re.compile(r"^/api/v1/datasets/(?P<id>dataset-[^/]+)/preview$"), "dataset"),
    (re.compile(r"^/api/v1/training-jobs/(?P<id>job-[^/]+)"), "training_job"),
    (re.compile(r"^/api/v1/evaluations/(?P<id>evaluation-[^/]+)$"), "evaluation"),
    (re.compile(r"^/api/v1/exports/(?P<id>export-[^/]+)$"), "export"),
)
_LIST_PATHS = {
    "/api/v1/models": "model",
    "/api/v1/training-jobs": "training_job",
}
_BIND_PATHS = {
    ("POST", "/api/v1/models"): ("model", "id"),
    ("POST", "/api/v1/datasets/upload"): ("dataset_file", "id"),
    ("POST", "/api/v1/datasets/inspect"): ("dataset", "snapshot_id"),
    ("POST", "/api/v1/training-jobs"): ("training_job", "id"),
    ("POST", "/api/v1/evaluations"): ("evaluation", "id"),
    ("POST", "/api/v1/exports"): ("export", "id"),
}
_SENSITIVE_ACTIONS = {
    ("POST", "/api/v1/compute-nodes/register"): "node.create",
    ("POST", "/api/v1/models"): "model.create",
    ("POST", "/api/v1/datasets/upload"): "dataset_file.create",
    ("POST", "/api/v1/datasets/inspect"): "dataset.create",
    ("POST", "/api/v1/training-jobs"): "training_job.create",
    ("POST", "/api/v1/evaluations"): "evaluation.create",
    ("POST", "/api/v1/exports"): "export.create",
}


class ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)


class MembershipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)
    role: Literal["admin", "trainer", "viewer", "agent"]


class QuotaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=200)
    max_total_jobs: int | None = Field(default=None, ge=0)
    max_concurrent_jobs: int | None = Field(default=None, ge=0)
    max_upload_bytes: int | None = Field(default=None, ge=0)
    max_log_bytes_per_minute: int | None = Field(default=None, ge=0)


class VisibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visibility: Literal["private", "project", "company"]


class DeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=500)


class RetentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    older_than_days: int = Field(ge=1, le=36500)


def _principal(request: Request) -> Principal:
    return request.state.principal


def _admin(principal: Principal) -> None:
    if principal.role != "admin":
        raise SecurityError("ADMIN_REQUIRED", "role", "Only administrators can perform this operation.")


def install_security(app: FastAPI, service: SecurityService) -> FastAPI:
    async def security_error_handler(_request: Request, error: SecurityError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content={"detail": error.to_dict()})

    app.add_exception_handler(SecurityError, security_error_handler)
    app.include_router(create_security_router(service))
    app.add_middleware(SecurityMiddleware, service=service)
    return app


def create_security_router(service: SecurityService) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["MVP security"])

    @router.get("/auth/me")
    async def me(request: Request) -> dict[str, Any]:
        return service.principal_dict(_principal(request))

    @router.post("/security/projects", status_code=201)
    async def create_project(request: Request, payload: ProjectRequest) -> dict[str, Any]:
        principal = _principal(request)
        _admin(principal)
        return service.create_project(payload.name, principal.user_id, principal.display_name)

    @router.put("/security/projects/{project_id}/members")
    async def add_member(project_id: str, request: Request, payload: MembershipRequest) -> dict[str, str]:
        principal = _principal(request)
        _admin(principal)
        if project_id != principal.project_id:
            raise SecurityError("PROJECT_ACCESS_DENIED", "project_id", "Administrators manage their current project.")
        return service.add_member(project_id, payload.user_id, payload.display_name, payload.role)

    @router.get("/security/credentials")
    async def list_credentials(request: Request) -> dict[str, Any]:
        principal = _principal(request)
        if principal.role not in {"admin", "trainer"}:
            raise SecurityError("ROLE_READ_ONLY", "role", "Role cannot access credential metadata.")
        return {"items": service.list_credentials(principal)}

    @router.put("/security/quotas")
    async def set_quota(request: Request, payload: QuotaRequest) -> dict[str, int]:
        principal = _principal(request)
        _admin(principal)
        values = {key: value for key, value in payload.model_dump(exclude={"user_id"}).items() if value is not None}
        return service.set_quota(str(principal.project_id), payload.user_id, values)

    @router.put("/security/objects/{object_type}/{object_id}/visibility")
    async def set_visibility(
        object_type: str, object_id: str, request: Request, payload: VisibilityRequest
    ) -> dict[str, Any]:
        return service.set_visibility(_principal(request), object_type, object_id, payload.visibility)

    @router.delete("/security/objects/{object_type}/{object_id}")
    async def delete_object(
        object_type: str, object_id: str, request: Request, payload: DeleteRequest
    ) -> dict[str, Any]:
        return service.soft_delete(_principal(request), object_type, object_id, payload.reason)

    @router.post("/security/retention/run")
    async def run_retention(request: Request, payload: RetentionRequest) -> dict[str, Any]:
        return service.apply_retention(_principal(request), older_than_days=payload.older_than_days)

    @router.get("/security/audit")
    async def list_audit(request: Request, limit: int = 1000) -> dict[str, Any]:
        return {"items": service.list_audit(_principal(request), limit=min(max(limit, 1), 1000))}

    return router


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforce OIDC identity, project visibility, quotas, and sensitive audit."""

    def __init__(self, app: Any, service: SecurityService):
        super().__init__(app)
        self.service = service

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.url.path.startswith("/api/v1"):
            return await call_next(request)
        body_bytes = await request.body()
        body = self._json_body(body_bytes)
        project_id = request.headers.get("X-Project-ID")
        principal = None
        decision = self._request_decision(request)
        try:
            principal = self.service.authenticate(request.headers.get("Authorization"), project_id)
            request.state.principal = principal
            decision = self._authorize(request, principal, body, len(body_bytes))
        except SecurityError as error:
            if decision.sensitive:
                self.service.audit(
                    principal,
                    action=decision.action,
                    resource_type=decision.resource_type,
                    resource_id=decision.resource_id,
                    outcome="DENIED",
                    status_code=error.status_code,
                    detail={"code": error.code, "requested_project_id": project_id},
                )
            return JSONResponse(status_code=error.status_code, content={"detail": error.to_dict()})

        response = await call_next(request)
        payload = None
        if self._needs_payload(request.method, request.url.path, response):
            response, payload = await self._capture_response(response)
        if 200 <= response.status_code < 300 and payload is not None:
            response, payload = self._post_process(request, response, payload, decision)
        if decision.sensitive:
            resource_id = decision.resource_id or self._response_resource_id(request.method, request.url.path, payload)
            self.service.audit(
                principal,
                action=decision.action,
                resource_type=decision.resource_type,
                resource_id=resource_id,
                outcome="SUCCESS" if response.status_code < 400 else "FAILED",
                status_code=response.status_code,
            )
        return response

    def _authorize(
        self, request: Request, principal: Principal, body: Mapping[str, Any], request_bytes: int
    ) -> RequestDecision:
        method, path = request.method, request.url.path
        decision = self._request_decision(request)
        decision = RequestDecision(
            principal,
            principal.project_id,
            decision.action,
            decision.resource_type,
            decision.resource_id,
            decision.sensitive,
        )
        if path.startswith("/api/v1/security/"):
            if method != "GET" or path != "/api/v1/security/credentials":
                _admin(principal)
            return decision
        agent_endpoint = path.startswith("/api/v1/agent/") or (
            "/heartbeat" in path and path.startswith("/api/v1/compute-nodes/")
        )
        if agent_endpoint:
            if principal.role not in {"admin", "agent"}:
                raise SecurityError("AGENT_ROLE_REQUIRED", "role", "Worker endpoint requires an Agent identity.")
            if path.endswith("/logs"):
                self.service.consume_log_quota(principal, request_bytes)
            return decision
        if principal.role == "agent":
            raise SecurityError("AGENT_SCOPE_DENIED", "role", "Agent identity is limited to Worker endpoints.")
        if method == "POST" and path == "/api/v1/compute-nodes/register":
            _admin(principal)
            return decision
        if principal.role == "viewer" and method not in {"GET", "HEAD"}:
            raise SecurityError("ROLE_READ_ONLY", "role", "Viewer role is read-only.")

        if method == "POST" and path == "/api/v1/datasets/upload":
            self.service.check_upload_quota(principal, request_bytes)
        if method == "POST" and path == "/api/v1/training-jobs":
            self.service.check_job_quota(principal)

        credential_id = body.get("credential_id")
        if credential_id and not self.service.credential_exists(principal, str(credential_id)):
            raise SecurityError("CREDENTIAL_ACCESS_DENIED", "credential_id", "Credential is not available.", 404)
        self._authorize_references(principal, method, path, body)

        resource = self._path_resource(path)
        if resource is not None:
            object_type, object_id = resource
            write = path.endswith(("/stop", "/resume"))
            if "/artifacts/" in path:
                write = False
            self.service.require_access(principal, object_type, object_id, write=write)
            decision = RequestDecision(
                principal,
                principal.project_id,
                decision.action,
                object_type,
                object_id,
                decision.sensitive,
            )
        return decision

    def _authorize_references(
        self, principal: Principal, method: str, path: str, body: Mapping[str, Any]
    ) -> None:
        if method != "POST":
            return
        references: list[tuple[str, str]] = []
        if path == "/api/v1/datasets/inspect":
            file_ids = body.get("dataset_file_ids") or [body.get("dataset_file_id")]
            references.extend(("dataset_file", str(file_id)) for file_id in file_ids if file_id)
        elif path == "/api/v1/training-jobs":
            references.extend(
                (("model", str(body.get("model_id"))), ("dataset", str(body.get("dataset_snapshot_id"))))
            )
        elif path == "/api/v1/chat/completions":
            references.append(("model", str(body.get("model_id"))))
            if body.get("artifact_id"):
                references.append(("training_job", self._artifact_job(str(body["artifact_id"]))))
        elif path == "/api/v1/evaluations":
            references.extend(
                (
                    ("model", str(body.get("model_id"))),
                    ("dataset", str(body.get("dataset_snapshot_id"))),
                    ("training_job", self._artifact_job(str(body.get("artifact_id")))),
                )
            )
        elif path == "/api/v1/exports":
            references.extend(
                (
                    ("model", str(body.get("model_id"))),
                    ("training_job", self._artifact_job(str(body.get("adapter_id")))),
                )
            )
        for object_type, object_id in references:
            self.service.require_access(principal, object_type, object_id)

    def _artifact_job(self, artifact_id: str) -> str:
        if self.service.training_service is None:
            raise SecurityError("ARTIFACT_ACCESS_DENIED", "artifact_id", "Artifact resolver is unavailable.", 404)
        try:
            return self.service.training_service.get_artifact(artifact_id)["job_id"]
        except ValueError as error:
            raise SecurityError("ARTIFACT_ACCESS_DENIED", "artifact_id", "Artifact is not available.", 404) from error

    def _post_process(
        self, request: Request, response: Response, payload: Any, decision: RequestDecision
    ) -> tuple[Response, Any]:
        bind = _BIND_PATHS.get((request.method, request.url.path))
        if request.url.path.endswith("/resume") and request.method == "POST":
            bind = ("training_job", "id")
        if bind and isinstance(payload, dict) and payload.get(bind[1]):
            self.service.bind_object(decision.principal, bind[0], str(payload[bind[1]]))
        object_type = _LIST_PATHS.get(request.url.path) if request.method == "GET" else None
        if object_type and isinstance(payload, dict) and isinstance(payload.get("items"), list):
            payload["items"] = self.service.filter_items(decision.principal, object_type, payload["items"])
            response = self._json_response(response, payload)
        return response, payload

    def _request_decision(self, request: Request) -> RequestDecision:
        decision = self._base_decision(request.method, request.url.path)
        resource = self._path_resource(request.url.path)
        if resource is not None:
            decision = RequestDecision(
                decision.principal,
                decision.project_id,
                decision.action,
                resource[0],
                resource[1],
                decision.sensitive,
            )
        if (
            request.method == "GET"
            and request.url.path.endswith("/logs")
            and request.query_params.get("download", "false").lower() == "true"
        ):
            return RequestDecision(
                decision.principal,
                decision.project_id,
                "log.download",
                "training_job",
                decision.resource_id,
                True,
            )
        return decision

    def _base_decision(self, method: str, path: str) -> RequestDecision:
        action = _SENSITIVE_ACTIONS.get((method, path), f"{method.lower()} {path}")
        resource_type = _SENSITIVE_ACTIONS.get((method, path), "").split(".", 1)[0] or None
        sensitive = (method, path) in _SENSITIVE_ACTIONS
        if method == "POST" and path.endswith("/stop"):
            action, resource_type, sensitive = "training_job.stop", "training_job", True
        elif method == "POST" and path.endswith("/resume"):
            action, resource_type, sensitive = "training_job.resume", "training_job", True
        elif method == "GET" and path.endswith("/download"):
            action, resource_type, sensitive = "artifact.download", "training_job", True
        elif method == "DELETE":
            action, resource_type, sensitive = "object.delete", "managed_object", True
        elif path == "/api/v1/security/projects" and method == "POST":
            action, resource_type, sensitive = "project.create", "project", True
        elif path.endswith("/members") and method == "PUT":
            action, resource_type, sensitive = "membership.update", "membership", True
        elif path == "/api/v1/security/quotas" and method == "PUT":
            action, resource_type, sensitive = "quota.update", "quota", True
        elif path == "/api/v1/security/retention/run" and method == "POST":
            action, resource_type, sensitive = "retention.run", "managed_object", True
        if method == "DELETE" and path.startswith("/api/v1/security/objects/"):
            parts = path.rsplit("/", 2)
            if len(parts) == 3:
                resource_type, resource_id = parts[-2], parts[-1]
            else:
                resource_id = None
        else:
            resource_id = None
        return RequestDecision(
            Principal("", "", "viewer", None), None, action, resource_type, resource_id, sensitive
        )

    @staticmethod
    def _path_resource(path: str) -> tuple[str, str] | None:
        for pattern, object_type in _RESOURCE_PATHS:
            match = pattern.match(path)
            if match:
                return object_type, match.group("id")
        return None

    @staticmethod
    def _json_body(body: bytes) -> Mapping[str, Any]:
        if not body:
            return {}
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _needs_payload(method: str, path: str, response: Response) -> bool:
        return response.status_code < 400 and (
            (method, path) in _BIND_PATHS
            or (method == "POST" and path.endswith("/resume"))
            or (method == "GET" and path in _LIST_PATHS)
        )

    @staticmethod
    async def _capture_response(response: Response) -> tuple[Response, Any]:
        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        headers = dict(response.headers)
        headers.pop("content-length", None)
        rebuilt = Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )
        return rebuilt, payload

    @staticmethod
    def _json_response(response: Response, payload: Any) -> Response:
        headers = dict(response.headers)
        headers.pop("content-length", None)
        return JSONResponse(payload, status_code=response.status_code, headers=headers, background=response.background)

    @staticmethod
    def _response_resource_id(method: str, path: str, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        bind = _BIND_PATHS.get((method, path))
        if path.endswith("/resume"):
            bind = ("training_job", "id")
        return str(payload.get(bind[1])) if bind and payload.get(bind[1]) else None
