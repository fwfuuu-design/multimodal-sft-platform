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

import io
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from sft_platform.api.assets import create_asset_app
from sft_platform.mvp.assets import DatasetAssetService, FakeHub, ModelAssetService


class MvpAssetApiContractTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        models = root / "models"
        datasets = root / "datasets"
        media = root / "media"
        models.mkdir()
        datasets.mkdir()
        media.mkdir()
        fake_hub = FakeHub(
            {
                ("huggingface", "company/text-model"): {
                    "private": True,
                    "modalities": ["text"],
                    "template": "qwen",
                    "context_length": 8192,
                }
            }
        )
        model_service = ModelAssetService([models], fake_hub)
        dataset_service = DatasetAssetService([datasets], [media])
        self.client = TestClient(create_asset_app(model_service, dataset_service))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_model_validate_register_list_get_and_registry_contracts(self):
        payload = {
            "name": "text-model",
            "source": "huggingface",
            "location": "company/text-model",
            "credential_id": "cred-private-hf",
        }
        validated = self.client.post("/api/v1/models/validate", json=payload)
        assert validated.status_code == 200
        assert validated.json()["compatibility"]["modes"] == ["text_sft"]
        assert "credential_id" not in validated.json()

        registered = self.client.post("/api/v1/models", json=payload)
        assert registered.status_code == 201
        model_id = registered.json()["id"]
        listed = self.client.get("/api/v1/models")
        assert listed.status_code == 200
        assert [model["id"] for model in listed.json()["items"]] == [model_id]
        assert self.client.get(f"/api/v1/models/{model_id}").json()["id"] == model_id

        registry = self.client.get("/api/v1/models/registry")
        assert registry.status_code == 200
        assert len(registry.json()["items"]) == 638

    def test_private_model_and_unknown_fields_return_stable_rejections(self):
        missing_credential = self.client.post(
            "/api/v1/models/validate",
            json={
                "name": "private",
                "source": "huggingface",
                "location": "company/text-model",
            },
        )
        assert missing_credential.status_code == 422
        assert missing_credential.json()["detail"]["code"] == "MODEL_CREDENTIAL_REQUIRED"

        extra_secret = self.client.post(
            "/api/v1/models/validate",
            json={
                "name": "private",
                "source": "huggingface",
                "location": "company/text-model",
                "token": "must-not-be-accepted",
            },
        )
        assert extra_secret.status_code == 422
        assert "must-not-be-accepted" not in extra_secret.text

    def test_dataset_upload_inspect_and_snapshot_preview_contracts(self):
        records = [
            {
                "instruction": "Say hello",
                "input": "",
                "output": "Hello",
            }
        ]
        uploaded = self.client.post(
            "/api/v1/datasets/upload",
            files={"file": ("train.json", io.BytesIO(json.dumps(records).encode()), "application/json")},
        )
        assert uploaded.status_code == 201
        file_id = uploaded.json()["id"]
        assert uploaded.json()["filename"] == "train.json"

        inspected = self.client.post(
            "/api/v1/datasets/inspect",
            json={"dataset_file_id": file_id, "formatting": "alpaca", "preview_limit": 10},
        )
        assert inspected.status_code == 200
        inspection = inspected.json()
        assert inspection["ready"] is True
        assert inspection["statistics"]["sample_count"] == 1

        preview = self.client.get(f"/api/v1/datasets/{inspection['snapshot_id']}/preview")
        assert preview.status_code == 200
        assert preview.json()["items"] == inspection["preview"]

        second_records = [{"instruction": "Say bye", "input": "", "output": "Bye"}]
        second_upload = self.client.post(
            "/api/v1/datasets/upload",
            files={
                "file": (
                    "train-2.jsonl",
                    io.BytesIO((json.dumps(second_records[0]) + "\n").encode()),
                    "application/jsonl",
                )
            },
        )
        combined = self.client.post(
            "/api/v1/datasets/inspect",
            json={
                "dataset_file_ids": [file_id, second_upload.json()["id"]],
                "formatting": "alpaca",
            },
        )
        assert combined.status_code == 200
        assert combined.json()["statistics"]["source_file_count"] == 2
        assert combined.json()["statistics"]["sample_count"] == 2
        assert combined.json()["source_path"] is None

    def test_dataset_upload_rejects_non_mvp_file_types(self):
        response = self.client.post(
            "/api/v1/datasets/upload",
            files={"file": ("train.csv", io.BytesIO(b"instruction,output"), "text/csv")},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "DATASET_FILE_TYPE_NOT_SUPPORTED"


if __name__ == "__main__":
    unittest.main()
