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

"""HTTP contracts for stage-5 result, evaluation, Chat, and Export flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..mvp.results import ResultService, ResultServiceError


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str = Field(max_length=100_000)


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"]
    asset_id: str = Field(min_length=1, max_length=1000)


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str | list[TextContent | ImageContent]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=100)
    artifact_id: str | None = Field(default=None, max_length=100)
    messages: list[ChatMessageRequest] = Field(min_length=1, max_length=1000)
    template: str | None = Field(default=None, max_length=200)
    tokenizer: str = Field(default="auto", min_length=1, max_length=200)
    image_processor: str | None = Field(default="auto", max_length=200)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=100)
    artifact_id: str = Field(min_length=1, max_length=100)
    dataset_snapshot_id: str = Field(min_length=1, max_length=100)
    template: str | None = Field(default=None, max_length=200)
    predict: bool = True


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=100)
    adapter_id: str = Field(min_length=1, max_length=100)
    node_id: str = Field(min_length=1, max_length=100)
    output_name: str = Field(min_length=1, max_length=200)


def _raise_http_error(error: ResultServiceError) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.to_dict()) from error


def create_result_router(service: ResultService) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["MVP result service"])

    @router.get("/training-jobs/{job_id}/manifest")
    async def training_manifest(job_id: str) -> dict[str, Any]:
        try:
            return service.training_manifest(job_id)
        except ResultServiceError as error:
            _raise_http_error(error)

    @router.get("/training-jobs/{job_id}/artifacts/{artifact_id}/download", response_model=None)
    async def download_adapter(job_id: str, artifact_id: str) -> FileResponse:
        try:
            artifact = service.downloadable_artifact(job_id, artifact_id)
            return FileResponse(
                artifact["path"],
                media_type="application/octet-stream",
                filename=Path(artifact["path"]).name,
            )
        except ResultServiceError as error:
            _raise_http_error(error)

    @router.post("/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        try:
            return service.chat(
                model_id=request.model_id,
                artifact_id=request.artifact_id,
                messages=[message.model_dump() for message in request.messages],
                template=request.template,
                tokenizer=request.tokenizer,
                image_processor=request.image_processor,
            )
        except ResultServiceError as error:
            _raise_http_error(error)

    @router.post("/evaluations", status_code=status.HTTP_201_CREATED)
    async def evaluate(request: EvaluationRequest) -> dict[str, Any]:
        try:
            return service.evaluate(
                model_id=request.model_id,
                artifact_id=request.artifact_id,
                dataset_snapshot_id=request.dataset_snapshot_id,
                template=request.template,
                predict=request.predict,
            )
        except ResultServiceError as error:
            _raise_http_error(error)

    @router.get("/evaluations/{evaluation_id}")
    async def get_evaluation(evaluation_id: str) -> dict[str, Any]:
        try:
            return service.get_evaluation(evaluation_id)
        except ResultServiceError as error:
            _raise_http_error(error)

    @router.post("/exports", status_code=status.HTTP_201_CREATED)
    async def export_adapter(request: ExportRequest) -> dict[str, Any]:
        try:
            return service.export_adapter(
                model_id=request.model_id,
                adapter_id=request.adapter_id,
                node_id=request.node_id,
                output_name=request.output_name,
            )
        except ResultServiceError as error:
            _raise_http_error(error)

    @router.get("/exports/{export_id}")
    async def get_export(export_id: str) -> dict[str, Any]:
        try:
            return service.get_export(export_id)
        except ResultServiceError as error:
            _raise_http_error(error)

    return router
