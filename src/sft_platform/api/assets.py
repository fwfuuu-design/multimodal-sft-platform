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

"""HTTP contracts for stage-3 model and dataset asset services."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..mvp.assets import (
    AssetValidationError,
    DatasetAssetService,
    ModelAssetService,
    ModelReference,
    build_upstream_model_catalog,
)


class ModelReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    source: str
    location: str = Field(min_length=1, max_length=1000)
    credential_id: str | None = Field(default=None, max_length=128)


class DatasetInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_file_id: str | None = Field(default=None, min_length=1, max_length=100)
    dataset_file_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    formatting: str
    mapping: dict[str, str] = Field(default_factory=dict)
    preview_limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_file_ids(self) -> DatasetInspectRequest:
        if (self.dataset_file_id is None) == (self.dataset_file_ids is None):
            raise ValueError("Provide exactly one of dataset_file_id or dataset_file_ids.")
        return self

    def resolved_file_ids(self) -> list[str]:
        if self.dataset_file_ids is not None:
            return self.dataset_file_ids
        assert self.dataset_file_id is not None
        return [self.dataset_file_id]


def _model_reference(request: ModelReferenceRequest) -> ModelReference:
    return ModelReference(
        name=request.name,
        source=request.source,
        location=request.location,
        credential_id=request.credential_id,
    )


def _raise_http_error(error: AssetValidationError) -> None:
    raise HTTPException(status_code=422, detail=error.to_dict()) from error


def create_asset_router(
    model_service: ModelAssetService,
    dataset_service: DatasetAssetService,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["MVP assets"])

    @router.post("/models/validate")
    async def validate_model(request: ModelReferenceRequest) -> dict[str, Any]:
        try:
            return model_service.validate(_model_reference(request))
        except AssetValidationError as error:
            _raise_http_error(error)

    @router.post("/models", status_code=status.HTTP_201_CREATED)
    async def register_model(request: ModelReferenceRequest) -> dict[str, Any]:
        try:
            return model_service.register(_model_reference(request))
        except AssetValidationError as error:
            _raise_http_error(error)

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        return {"items": model_service.list()}

    @router.get("/models/registry")
    async def list_upstream_registry() -> dict[str, Any]:
        return {"items": build_upstream_model_catalog()}

    @router.get("/models/{model_id}")
    async def get_model(model_id: str) -> dict[str, Any]:
        try:
            return model_service.get(model_id)
        except AssetValidationError as error:
            _raise_http_error(error)

    @router.post("/datasets/upload", status_code=status.HTTP_201_CREATED)
    async def upload_dataset(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        try:
            return dataset_service.store_upload(file.filename or "dataset", file.file)
        except AssetValidationError as error:
            _raise_http_error(error)
        finally:
            await file.close()

    @router.post("/datasets/inspect")
    async def inspect_dataset(request: DatasetInspectRequest) -> dict[str, Any]:
        try:
            return dataset_service.inspect_uploads(
                request.resolved_file_ids(),
                request.formatting,
                request.mapping,
                request.preview_limit,
            ).to_dict()
        except AssetValidationError as error:
            _raise_http_error(error)

    @router.get("/datasets/{snapshot_id}/preview")
    async def get_dataset_preview(
        snapshot_id: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        try:
            records = dataset_service.get_snapshot(snapshot_id)
            return {"snapshot_id": snapshot_id, "items": records[:limit], "total": len(records)}
        except AssetValidationError as error:
            _raise_http_error(error)

    return router


def create_asset_app(
    model_service: ModelAssetService,
    dataset_service: DatasetAssetService,
) -> FastAPI:
    """Create a no-model metadata API app for contract tests and later service composition."""
    app = FastAPI(title="Multimodal Fine-tuning MVP Assets", version="1.0.0")

    @app.exception_handler(RequestValidationError)
    async def sanitized_validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        sanitized_errors = []
        for item in error.errors():
            sanitized_errors.append({key: value for key, value in item.items() if key not in {"input", "ctx"}})
        return JSONResponse(status_code=422, content={"detail": sanitized_errors})

    app.include_router(create_asset_router(model_service, dataset_service))
    return app
