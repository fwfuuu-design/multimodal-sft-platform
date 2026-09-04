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

import json
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from sft_platform.api.jobs import MVP_API_VERSION
from sft_platform.api.mvp_runtime import (
    RuntimeConfigurationError,
    RuntimeSettings,
    bootstrap_runtime_project,
    create_runtime_app,
)
from sft_platform.mvp.assets import FakeHub
from sft_platform.mvp.results import FakeModelAdapter
from sft_platform.mvp.security import FakeOIDCVerifier, Identity, SecurityError


def token_verifier_factory():
    return FakeOIDCVerifier({"test-token": Identity("test-user", "Test User")})


def hub_inspector_factory():
    return FakeHub()


def model_adapter_factory():
    return FakeModelAdapter()


def invalid_plugin_factory():
    return object()


class TestMvpRuntime:
    def setup_method(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("state", "models", "data", "media", "results"):
            (self.root / name).mkdir()
        self.environment = {
            "MVP_DATABASE_PATH": str(self.root / "state" / "service.sqlite3"),
            "MVP_STATE_ROOT": str(self.root / "state"),
            "MVP_MODEL_ROOTS": str(self.root / "models"),
            "MVP_DATA_ROOTS": str(self.root / "data"),
            "MVP_MEDIA_ROOTS": str(self.root / "media"),
            "MVP_RESULTS_ROOT": str(self.root / "results"),
            "MVP_FERNET_KEY": Fernet.generate_key().decode(),
            "MVP_TOKEN_VERIFIER_FACTORY": "tests.test_mvp_runtime:token_verifier_factory",
            "MVP_HUB_INSPECTOR_FACTORY": "tests.test_mvp_runtime:hub_inspector_factory",
            "MVP_MODEL_ADAPTER_FACTORY": "tests.test_mvp_runtime:model_adapter_factory",
        }

    def teardown_method(self):
        self.temporary.cleanup()

    def test_runtime_requires_every_security_and_adapter_integration(self):
        for field in (
            "MVP_FERNET_KEY",
            "MVP_TOKEN_VERIFIER_FACTORY",
            "MVP_HUB_INSPECTOR_FACTORY",
            "MVP_MODEL_ADAPTER_FACTORY",
        ):
            environment = {**self.environment, field: ""}
            with pytest.raises(RuntimeConfigurationError, match=field):
                RuntimeSettings.from_environment(environment)

        environment = {
            **self.environment,
            "MVP_TOKEN_VERIFIER_FACTORY": "tests.test_mvp_runtime:invalid_plugin_factory",
        }
        with pytest.raises(RuntimeConfigurationError, match="missing methods: verify"):
            create_runtime_app(environment)

    def test_runtime_exposes_only_the_authenticated_versioned_mvp_contract(self):
        app = create_runtime_app(self.environment)
        assert app.state.mvp_security_enabled is True
        assert app.state.api_contract_version == MVP_API_VERSION
        assert app.state.database_schema_version == 1
        assert "fernet" not in str(app.state.runtime_configuration).lower()

        schema = app.openapi()
        assert schema["info"]["version"] == MVP_API_VERSION
        assert schema["x-api-contract-version"] == MVP_API_VERSION
        assert schema["x-project-context-header"] == "X-Project-ID"
        assert schema["components"]["securitySchemes"]["OIDCBearer"]["scheme"] == "bearer"
        assert schema["security"] == [{"OIDCBearer": []}]
        assert "/api/v1/training-jobs" in schema["paths"]
        assert "/api/v1/chat/completions" in schema["paths"]
        assert "/v1/chat/completions" not in schema["paths"]

        with TestClient(app) as client:
            unauthorized = client.get("/api/v1/auth/me")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["detail"]["code"] == "AUTH_REQUIRED"

    def test_openapi_matches_the_frozen_stage_7_contract(self):
        expected_path = Path(__file__).resolve().parents[1] / "docs" / "openapi-v1.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        assert create_runtime_app(self.environment).openapi() == expected

    def test_initial_project_bootstrap_is_local_audited_and_exactly_once(self):
        environment = {
            **self.environment,
            "MVP_BOOTSTRAP_PROJECT_NAME": "Internal Platform",
            "MVP_BOOTSTRAP_ADMIN_SUBJECT": "test-user",
            "MVP_BOOTSTRAP_ADMIN_DISPLAY_NAME": "Initial Admin",
        }
        result = bootstrap_runtime_project(environment)
        assert result["created"] is True
        assert result["admin_user_id"] == "test-user"

        app = create_runtime_app(self.environment)
        with TestClient(app) as client:
            me = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer test-token"})
            audit = client.get(
                "/api/v1/security/audit",
                headers={"Authorization": "Bearer test-token"},
            )
        assert me.status_code == 200
        assert me.json()["role"] == "admin"
        assert audit.status_code == 200
        assert "project.bootstrap" in {item["action"] for item in audit.json()["items"]}

        with pytest.raises(SecurityError, match="BOOTSTRAP_ALREADY_COMPLETED"):
            bootstrap_runtime_project(environment)

    def test_invalid_key_and_uncontrolled_roots_fail_before_server_start(self):
        environment = {**self.environment, "MVP_FERNET_KEY": "not-a-fernet-key"}
        with pytest.raises(ValueError, match="valid Fernet master key"):
            create_runtime_app(environment)

        environment = {**self.environment, "MVP_MODEL_ROOTS": str(self.root / "missing")}
        with pytest.raises(RuntimeConfigurationError, match="MVP_MODEL_ROOTS"):
            create_runtime_app(environment)
