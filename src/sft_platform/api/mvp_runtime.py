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

"""Fail-closed runtime composition for the authenticated MVP control plane.

Production integrations are loaded through explicit factories. There is no
fallback to Fake OIDC, Fake Hub, or Fake Model Adapter in this entry point.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..mvp.assets import DatasetAssetService, ModelAssetService
from ..mvp.jobs import TrainingJobService
from ..mvp.operations import prepare_database_schema
from ..mvp.results import ResultService
from ..mvp.security import SecurityService
from .jobs import create_secure_mvp_app


_REQUIRED_ENVIRONMENT = (
    "MVP_DATABASE_PATH",
    "MVP_STATE_ROOT",
    "MVP_MODEL_ROOTS",
    "MVP_DATA_ROOTS",
    "MVP_MEDIA_ROOTS",
    "MVP_RESULTS_ROOT",
    "MVP_FERNET_KEY",
    "MVP_TOKEN_VERIFIER_FACTORY",
    "MVP_HUB_INSPECTOR_FACTORY",
    "MVP_MODEL_ADAPTER_FACTORY",
)


class RuntimeConfigurationError(ValueError):
    """Raised before startup when a secure runtime dependency is absent."""


@dataclass(frozen=True)
class RuntimeSettings:
    database_path: Path
    state_root: Path
    model_roots: tuple[Path, ...]
    data_roots: tuple[Path, ...]
    media_roots: tuple[Path, ...]
    results_root: Path
    fernet_key: bytes
    token_verifier_factory: str
    hub_inspector_factory: str
    model_adapter_factory: str

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> RuntimeSettings:
        env = os.environ if environment is None else environment
        missing = [name for name in _REQUIRED_ENVIRONMENT if not env.get(name)]
        if missing:
            raise RuntimeConfigurationError(f"Missing required runtime configuration: {', '.join(missing)}")

        state_root = _directory(env["MVP_STATE_ROOT"], "MVP_STATE_ROOT", create=True)
        results_root = _directory(env["MVP_RESULTS_ROOT"], "MVP_RESULTS_ROOT", create=True)
        database_path = Path(env["MVP_DATABASE_PATH"]).expanduser().resolve()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        if database_path.exists() and not database_path.is_file():
            raise RuntimeConfigurationError("MVP_DATABASE_PATH must point to a file.")

        return cls(
            database_path=database_path,
            state_root=state_root,
            model_roots=_directory_list(env["MVP_MODEL_ROOTS"], "MVP_MODEL_ROOTS"),
            data_roots=_directory_list(env["MVP_DATA_ROOTS"], "MVP_DATA_ROOTS"),
            media_roots=_directory_list(env["MVP_MEDIA_ROOTS"], "MVP_MEDIA_ROOTS"),
            results_root=results_root,
            fernet_key=env["MVP_FERNET_KEY"].encode(),
            token_verifier_factory=env["MVP_TOKEN_VERIFIER_FACTORY"],
            hub_inspector_factory=env["MVP_HUB_INSPECTOR_FACTORY"],
            model_adapter_factory=env["MVP_MODEL_ADAPTER_FACTORY"],
        )

    def public_summary(self) -> dict[str, Any]:
        """Return non-secret startup metadata suitable for diagnostics."""
        return {
            "database_path": str(self.database_path),
            "state_root": str(self.state_root),
            "model_roots": [str(path) for path in self.model_roots],
            "data_roots": [str(path) for path in self.data_roots],
            "media_roots": [str(path) for path in self.media_roots],
            "results_root": str(self.results_root),
            "token_verifier_factory": self.token_verifier_factory,
            "hub_inspector_factory": self.hub_inspector_factory,
            "model_adapter_factory": self.model_adapter_factory,
        }


def _directory(value: str, field: str, *, create: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeConfigurationError(f"{field} must reference an existing directory: {path}")
    return path


def _directory_list(value: str, field: str) -> tuple[Path, ...]:
    items = tuple(_directory(item, field) for item in value.split(os.pathsep) if item)
    if not items:
        raise RuntimeConfigurationError(f"{field} must contain at least one directory.")
    return items


def _load_plugin(factory_path: str, field: str, required_methods: tuple[str, ...]) -> Any:
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise RuntimeConfigurationError(f"{field} must use the 'module:factory' format.")
    try:
        factory = getattr(importlib.import_module(module_name), attribute_name)
        plugin = factory()
    except Exception as error:
        raise RuntimeConfigurationError(f"Unable to initialize {field}.") from error
    missing = [method for method in required_methods if not callable(getattr(plugin, method, None))]
    if missing:
        raise RuntimeConfigurationError(f"{field} plugin is missing methods: {', '.join(missing)}")
    return plugin


def create_runtime_app(environment: Mapping[str, str] | None = None):
    """Build the only supported deployment app; startup fails on missing integrations."""
    settings = RuntimeSettings.from_environment(environment)
    schema_version = prepare_database_schema(settings.database_path)
    token_verifier = _load_plugin(settings.token_verifier_factory, "MVP_TOKEN_VERIFIER_FACTORY", ("verify",))
    hub_inspector = _load_plugin(settings.hub_inspector_factory, "MVP_HUB_INSPECTOR_FACTORY", ("inspect",))
    model_adapter = _load_plugin(
        settings.model_adapter_factory,
        "MVP_MODEL_ADAPTER_FACTORY",
        ("complete", "predict"),
    )

    model_service = ModelAssetService(settings.model_roots, hub_inspector, settings.state_root)
    dataset_service = DatasetAssetService(settings.data_roots, settings.media_roots, settings.state_root)
    training_service = TrainingJobService(
        settings.database_path,
        model_service=model_service,
        dataset_service=dataset_service,
    )
    result_service = ResultService(
        settings.database_path,
        settings.results_root,
        model_service=model_service,
        dataset_service=dataset_service,
        training_service=training_service,
        model_adapter=model_adapter,
    )
    security_service = SecurityService(
        settings.database_path,
        token_verifier=token_verifier,
        encryption_key=settings.fernet_key,
        training_service=training_service,
    )
    app = create_secure_mvp_app(
        model_service,
        dataset_service,
        training_service,
        result_service,
        security_service,
    )
    app.state.runtime_configuration = settings.public_summary()
    app.state.database_schema_version = schema_version
    app.state.security_service = security_service
    return app


def bootstrap_runtime_project(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Create the first project through a local, one-time deployment operation."""
    env = os.environ if environment is None else environment
    required = ("MVP_BOOTSTRAP_PROJECT_NAME", "MVP_BOOTSTRAP_ADMIN_SUBJECT")
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise RuntimeConfigurationError(f"Missing bootstrap configuration: {', '.join(missing)}")
    app = create_runtime_app(env)
    return app.state.security_service.bootstrap_project(
        env["MVP_BOOTSTRAP_PROJECT_NAME"],
        env["MVP_BOOTSTRAP_ADMIN_SUBJECT"],
        env.get("MVP_BOOTSTRAP_ADMIN_DISPLAY_NAME"),
    )


def run_mvp_bootstrap() -> None:
    """Print the non-secret result of the one-time initial project bootstrap."""
    print(json.dumps(bootstrap_runtime_project(), ensure_ascii=False, sort_keys=True))


def run_mvp_api() -> None:
    """Launch the authenticated control plane from environment configuration."""
    import uvicorn

    host = os.getenv("MVP_API_HOST", "127.0.0.1")
    port = int(os.getenv("MVP_API_PORT", "8000"))
    uvicorn.run(create_runtime_app(), host=host, port=port)
