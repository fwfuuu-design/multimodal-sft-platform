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
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from sft_platform.api.jobs import create_mvp_app
from sft_platform.mvp.assets import DatasetAssetService, FakeHub, ModelAssetService, ModelReference
from sft_platform.mvp.jobs import (
    ComputeNodeRegistration,
    FakeWorkerAgent,
    GpuReport,
    JobSubmission,
    TrainingJobService,
)
from sft_platform.mvp.results import ResultService


class MvpResultApiContractTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        models_root = root / "models"
        datasets_root = root / "datasets"
        media_root = root / "media"
        outputs_root = root / "outputs"
        results_root = root / "results"
        for path in (models_root, datasets_root, media_root, outputs_root, results_root):
            path.mkdir()
        Image.new("RGB", (8, 8), "red").save(media_root / "sample.png", format="PNG")
        model_service = ModelAssetService(
            [models_root],
            FakeHub(
                {
                    ("huggingface", "fake/api-image-model"): {
                        "parameter_count": 1_000,
                        "estimated_model_bytes": 2_000,
                        "weight_precision": "fp16",
                        "template": "qwen3_vl",
                        "context_length": 8192,
                        "modalities": ["text", "image"],
                    }
                }
            ),
        )
        model = model_service.register(ModelReference("api-image", "huggingface", "fake/api-image-model"))
        dataset_service = DatasetAssetService([datasets_root], [media_root])
        dataset_path = datasets_root / "image.json"
        dataset_path.write_text(
            json.dumps(
                [
                    {
                        "instruction": "Describe <image>",
                        "input": "",
                        "output": "red",
                        "images": ["sample.png"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        dataset_id = dataset_service.inspect(dataset_path, "alpaca").snapshot_id
        training_service = TrainingJobService(
            root / "service.sqlite3", model_service=model_service, dataset_service=dataset_service
        )
        node = training_service.register_node(
            ComputeNodeRegistration(
                "api-node",
                "cloud",
                "https://worker.example.test/agent",
                str(models_root),
                str(datasets_root),
                str(outputs_root),
                {},
            )
        )
        self.node_id = node["id"]
        training_service.heartbeat(
            self.node_id,
            [GpuReport(0, "Fake GPU", 100_000, 100_000)],
            disk_free_bytes=100_000,
            system_memory_free_bytes=100_000,
            driver_version="fake-driver",
        )
        job = training_service.create_job(
            JobSubmission(
                "api qlora",
                model["id"],
                dataset_id,
                {
                    "stage": "sft",
                    "finetuning_type": "lora",
                    "template": "qwen3_vl",
                    "quantization_bit": 4,
                    "quantization_method": "bnb",
                },
                node_id=self.node_id,
            )
        )
        FakeWorkerAgent(training_service, self.node_id).run_next()
        artifacts = training_service.list_artifacts(job["id"])
        self.model_id = model["id"]
        self.dataset_id = dataset_id
        self.job_id = job["id"]
        self.adapter_id = next(item["id"] for item in artifacts if item["kind"] == "adapter")
        self.checkpoint_id = next(item["id"] for item in artifacts if item["kind"] == "checkpoint")
        result_service = ResultService(
            root / "service.sqlite3",
            results_root,
            model_service=model_service,
            dataset_service=dataset_service,
            training_service=training_service,
        )
        self.client = TestClient(create_mvp_app(model_service, dataset_service, training_service, result_service))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_manifest_chat_evaluation_prediction_and_export_contracts(self):
        manifest = self.client.get(f"/api/v1/training-jobs/{self.job_id}/manifest")
        assert manifest.status_code == 200
        assert manifest.json()["complete"] is True
        downloaded_adapter = self.client.get(
            f"/api/v1/training-jobs/{self.job_id}/artifacts/{self.adapter_id}/download"
        )
        assert downloaded_adapter.status_code == 200
        assert downloaded_adapter.headers["content-disposition"].endswith('"adapter_model.fake.safetensors"')
        assert b"No model weights are present" in downloaded_adapter.content

        base_chat = self.client.post(
            "/api/v1/chat/completions",
            json={
                "model_id": self.model_id,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert base_chat.status_code == 200
        assert base_chat.json()["artifact_id"] is None
        image_chat = self.client.post(
            "/api/v1/chat/completions",
            json={
                "model_id": self.model_id,
                "artifact_id": self.adapter_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "asset_id": "image-fake"},
                            {"type": "text", "text": "Describe"},
                        ],
                    }
                ],
            },
        )
        assert image_chat.status_code == 200
        assert image_chat.json()["modality"] == "image_text"

        evaluation = self.client.post(
            "/api/v1/evaluations",
            json={
                "model_id": self.model_id,
                "artifact_id": self.checkpoint_id,
                "dataset_snapshot_id": self.dataset_id,
                "predict": True,
            },
        )
        assert evaluation.status_code == 201
        evaluation_id = evaluation.json()["id"]
        assert self.client.get(f"/api/v1/evaluations/{evaluation_id}").json()["metrics"]["sample_count"] == 1

        exported = self.client.post(
            "/api/v1/exports",
            json={
                "model_id": self.model_id,
                "adapter_id": self.adapter_id,
                "node_id": self.node_id,
                "output_name": "api-merged-model",
            },
        )
        assert exported.status_code == 201, exported.text
        export_id = exported.json()["id"]
        assert self.client.get(f"/api/v1/exports/{export_id}").json()["state"] == "SUCCEEDED"

    def test_missing_adapter_and_non_merge_export_fields_are_rejected_without_fallback(self):
        missing = self.client.post(
            "/api/v1/chat/completions",
            json={
                "model_id": self.model_id,
                "artifact_id": "artifact-missing",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "ARTIFACT_NOT_FOUND"

        unsupported = self.client.post(
            "/api/v1/exports",
            json={
                "model_id": self.model_id,
                "adapter_id": self.adapter_id,
                "node_id": self.node_id,
                "output_name": "forbidden",
                "export_quantization_bit": "must-not-be-reflected",
                "ollama_modelfile": True,
            },
        )
        assert unsupported.status_code == 422
        assert "must-not-be-reflected" not in unsupported.text


if __name__ == "__main__":
    unittest.main()
