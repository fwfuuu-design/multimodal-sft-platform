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

"""HTTP contracts for compute nodes, Worker Agents, and training jobs."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from ..mvp.assets import DatasetAssetService, ModelAssetService
from ..mvp.jobs import (
    ComputeNodeRegistration,
    GpuReport,
    JobServiceError,
    JobSubmission,
    TrainingJobService,
)


MVP_API_VERSION = "1.0.0"
MVP_API_PREFIX = "/api/v1"


class NodeRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    node_type: Literal["local", "cloud"]
    endpoint: str = Field(min_length=1, max_length=1000)
    model_root: str = Field(min_length=1, max_length=2000)
    data_root: str = Field(min_length=1, max_length=2000)
    output_root: str = Field(min_length=1, max_length=2000)
    labels: dict[str, str] = Field(default_factory=dict)


class GpuReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    model: str = Field(min_length=1, max_length=200)
    memory_total_bytes: int = Field(ge=1)
    memory_free_bytes: int = Field(ge=0)
    utilization: float = Field(default=0.0, ge=0, le=1)


class NodeHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpus: list[GpuReportRequest] = Field(max_length=128)
    disk_free_bytes: int = Field(ge=0)
    system_memory_free_bytes: int | None = Field(default=None, ge=0)
    driver_version: str = Field(min_length=1, max_length=200)
    active_attempt_ids: list[str] = Field(default_factory=list, max_length=256)


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    model_id: str = Field(min_length=1, max_length=100)
    dataset_snapshot_id: str = Field(min_length=1, max_length=100)
    training_parameters: dict[str, Any]
    gpu_count: int = Field(default=1, ge=1, le=128)
    node_id: str | None = Field(default=None, max_length=100)
    minimum_disk_bytes: int = Field(default=0, ge=0)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str | None = Field(default=None, max_length=100)


class AgentPollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=100)


class AgentStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=100)
    state: Literal["RUNNING", "SUCCEEDED", "FAILED", "STOPPED"]
    metrics: dict[str, Any] | None = None
    error_category: str | None = Field(default=None, max_length=100)
    error_code: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=4000)


class LogEntryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream: Literal["stdout", "stderr", "system"] = "stdout"
    message: str = Field(max_length=64 * 1024)


class AgentLogsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=100)
    entries: list[LogEntryRequest] = Field(min_length=1, max_length=1000)


class ArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["adapter", "checkpoint", "config", "log", "metrics", "other"]
    path: str = Field(min_length=1, max_length=4000)
    bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    restorable: bool = False


class AgentArtifactsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=100)
    artifacts: list[ArtifactRequest] = Field(min_length=1, max_length=1000)


def _raise_http_error(error: JobServiceError) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.to_dict()) from error


def sanitized_validation_handler(_request: Request, error: RequestValidationError) -> JSONResponse:
    sanitized_errors = []
    for item in error.errors():
        sanitized_errors.append({key: value for key, value in item.items() if key not in {"input", "ctx"}})
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


def create_training_router(service: TrainingJobService) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["MVP training service"])

    @router.post("/compute-nodes/register", status_code=status.HTTP_201_CREATED)
    async def register_node(request: NodeRegistrationRequest) -> dict[str, Any]:
        try:
            return service.register_node(
                ComputeNodeRegistration(
                    name=request.name,
                    node_type=request.node_type,
                    endpoint=request.endpoint,
                    model_root=request.model_root,
                    data_root=request.data_root,
                    output_root=request.output_root,
                    labels=request.labels,
                )
            )
        except JobServiceError as error:
            _raise_http_error(error)

    @router.get("/compute-nodes")
    async def list_nodes() -> dict[str, Any]:
        return {"items": service.list_nodes()}

    @router.get("/compute-nodes/{node_id}")
    async def get_node(node_id: str) -> dict[str, Any]:
        try:
            return service.get_node(node_id)
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/compute-nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str, request: NodeHeartbeatRequest) -> dict[str, Any]:
        try:
            return service.heartbeat(
                node_id,
                [
                    GpuReport(
                        id=gpu.id,
                        model=gpu.model,
                        memory_total_bytes=gpu.memory_total_bytes,
                        memory_free_bytes=gpu.memory_free_bytes,
                        utilization=gpu.utilization,
                    )
                    for gpu in request.gpus
                ],
                disk_free_bytes=request.disk_free_bytes,
                driver_version=request.driver_version,
                system_memory_free_bytes=request.system_memory_free_bytes,
                active_attempt_ids=request.active_attempt_ids,
            )
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/training-jobs", status_code=status.HTTP_201_CREATED)
    async def create_job(request: JobCreateRequest) -> dict[str, Any]:
        try:
            return service.create_job(
                JobSubmission(
                    name=request.name,
                    model_id=request.model_id,
                    dataset_snapshot_id=request.dataset_snapshot_id,
                    training_parameters=request.training_parameters,
                    gpu_count=request.gpu_count,
                    node_id=request.node_id,
                    minimum_disk_bytes=request.minimum_disk_bytes,
                )
            )
        except JobServiceError as error:
            _raise_http_error(error)

    @router.get("/training-jobs")
    async def list_jobs(
        state_filter: Annotated[str | None, Query(alias="state")] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        try:
            return {"items": service.list_jobs(state=state_filter, limit=limit)}
        except JobServiceError as error:
            _raise_http_error(error)

    @router.get("/training-jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return service.get_job(job_id)
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/training-jobs/{job_id}/stop")
    async def stop_job(job_id: str) -> dict[str, Any]:
        try:
            return service.stop_job(job_id)
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/training-jobs/{job_id}/resume", status_code=status.HTTP_201_CREATED)
    async def resume_job(job_id: str, request: ResumeRequest) -> dict[str, Any]:
        try:
            return service.resume_job(job_id, request.checkpoint_id)
        except JobServiceError as error:
            _raise_http_error(error)

    @router.get("/training-jobs/{job_id}/events")
    async def get_events(
        job_id: str,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
    ) -> dict[str, Any]:
        try:
            return service.get_events(job_id, after_sequence=after_sequence, limit=limit)
        except JobServiceError as error:
            _raise_http_error(error)

    @router.get("/training-jobs/{job_id}/logs", response_model=None)
    async def get_logs(
        job_id: str,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 1000,
        download: bool = False,
    ) -> dict[str, Any] | PlainTextResponse:
        try:
            result = service.get_logs(job_id, after_sequence=after_sequence, limit=limit)
            if download:
                body = "\n".join(f"[{item['stream']}] {item['message']}" for item in result["items"])
                return PlainTextResponse(
                    body + ("\n" if body else ""),
                    headers={"Content-Disposition": f'attachment; filename="{job_id}.log"'},
                )
            return result
        except JobServiceError as error:
            _raise_http_error(error)

    @router.get("/training-jobs/{job_id}/artifacts")
    async def list_artifacts(job_id: str) -> dict[str, Any]:
        try:
            return {"items": service.list_artifacts(job_id)}
        except JobServiceError as error:
            _raise_http_error(error)

    # Worker Agent protocol. These endpoints are structured and contain no shell input.
    @router.post("/agent/tasks/poll")
    async def poll_task(request: AgentPollRequest) -> dict[str, Any]:
        try:
            return {"assignment": service.poll_assignment(request.node_id)}
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/agent/tasks/{attempt_id}/state")
    async def report_state(attempt_id: str, request: AgentStateRequest) -> dict[str, Any]:
        try:
            return service.report_state(
                request.node_id,
                attempt_id,
                request.state,
                metrics=request.metrics,
                error_category=request.error_category,
                error_code=request.error_code,
                error_message=request.error_message,
            )
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/agent/tasks/{attempt_id}/logs")
    async def append_logs(attempt_id: str, request: AgentLogsRequest) -> dict[str, Any]:
        try:
            return service.append_logs(
                request.node_id, attempt_id, [entry.model_dump() for entry in request.entries]
            )
        except JobServiceError as error:
            _raise_http_error(error)

    @router.post("/agent/tasks/{attempt_id}/artifacts")
    async def add_artifacts(attempt_id: str, request: AgentArtifactsRequest) -> dict[str, Any]:
        try:
            return {
                "items": service.add_artifacts(
                    request.node_id, attempt_id, [artifact.model_dump() for artifact in request.artifacts]
                )
            }
        except JobServiceError as error:
            _raise_http_error(error)

    return router


def create_training_app(service: TrainingJobService) -> FastAPI:
    app = FastAPI(title="Multimodal Fine-tuning MVP Training Service", version=MVP_API_VERSION)
    app.add_exception_handler(RequestValidationError, sanitized_validation_handler)
    app.include_router(create_training_router(service))
    return app


def create_mvp_app(
    model_service: ModelAssetService,
    dataset_service: DatasetAssetService,
    training_service: TrainingJobService,
    result_service: Any | None = None,
) -> FastAPI:
    """Compose stage-3 assets and stage-4 jobs without initializing a model."""
    from .assets import create_asset_router

    app = FastAPI(title="Multimodal Fine-tuning MVP Service", version=MVP_API_VERSION)
    app.state.mvp_security_enabled = False
    app.add_exception_handler(RequestValidationError, sanitized_validation_handler)
    app.include_router(create_asset_router(model_service, dataset_service))
    app.include_router(create_training_router(training_service))
    if result_service is not None:
        from .results import create_result_router

        app.include_router(create_result_router(result_service))
    return app


def create_secure_mvp_app(
    model_service: ModelAssetService,
    dataset_service: DatasetAssetService,
    training_service: TrainingJobService,
    result_service: Any,
    security_service: Any,
) -> FastAPI:
    """Compose the formal authenticated stage-6 API surface."""
    from .security import install_security

    app = create_mvp_app(model_service, dataset_service, training_service, result_service)
    install_security(app, security_service)
    app.state.mvp_security_enabled = True
    app.state.api_contract_version = MVP_API_VERSION

    def secure_openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            security_schemes = schema.setdefault("components", {}).setdefault("securitySchemes", {})
            security_schemes["OIDCBearer"] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "OIDC JWT",
                "description": "Company OIDC access token. X-Project-ID is required for users with multiple projects.",
            }
            schema["security"] = [{"OIDCBearer": []}]
            schema["x-api-contract-version"] = MVP_API_VERSION
            schema["x-project-context-header"] = "X-Project-ID"
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = secure_openapi
    return app
