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

from PIL import Image

from sft_platform.mvp.assets import DatasetAssetService, FakeHub, ModelAssetService, ModelReference
from sft_platform.mvp.jobs import (
    ComputeNodeRegistration,
    FakeWorkerAgent,
    GpuReport,
    JobSubmission,
    TrainingJobService,
)
from sft_platform.mvp.results import ResultService, ResultServiceError


class MvpResultServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.models_root = self.root / "models"
        self.datasets_root = self.root / "datasets"
        self.media_root = self.root / "media"
        self.outputs_root = self.root / "outputs"
        self.result_root = self.root / "results"
        for path in (
            self.models_root,
            self.datasets_root,
            self.media_root,
            self.outputs_root,
            self.result_root,
        ):
            path.mkdir()
        Image.new("RGB", (16, 12), "blue").save(self.media_root / "sample.png", format="PNG")

        self.hub = FakeHub(
            {
                ("huggingface", "fake/text-model"): {
                    "revision": "text-revision",
                    "parameter_count": 1_000,
                    "estimated_model_bytes": 2_000,
                    "weight_precision": "fp16",
                    "template": "qwen",
                    "context_length": 8192,
                    "modalities": ["text"],
                },
                ("modelscope", "fake/image-model"): {
                    "revision": "image-revision",
                    "parameter_count": 2_000,
                    "estimated_model_bytes": 4_000,
                    "weight_precision": "bf16",
                    "template": "qwen3_vl",
                    "context_length": 16384,
                    "modalities": ["text", "image"],
                },
            }
        )
        self.model_service = ModelAssetService([self.models_root], self.hub, state_root=self.root / "asset-state")
        self.text_model = self.model_service.register(
            ModelReference("text", "huggingface", "fake/text-model")
        )
        self.image_model = self.model_service.register(
            ModelReference("image", "modelscope", "fake/image-model")
        )
        self.dataset_service = DatasetAssetService(
            [self.datasets_root], [self.media_root], state_root=self.root / "asset-state"
        )
        text_path = self.datasets_root / "text.json"
        text_path.write_text(
            json.dumps([{"instruction": "Say hello", "input": "", "output": "Hello"}]), encoding="utf-8"
        )
        image_path = self.datasets_root / "image.json"
        image_path.write_text(
            json.dumps(
                [
                    {
                        "instruction": "Describe <image>",
                        "input": "",
                        "output": "Blue",
                        "images": ["sample.png"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.text_dataset = self.dataset_service.inspect(text_path, "alpaca").snapshot_id
        self.image_dataset = self.dataset_service.inspect(image_path, "alpaca").snapshot_id

        self.training_service = TrainingJobService(
            self.root / "service.sqlite3",
            model_service=self.model_service,
            dataset_service=self.dataset_service,
        )
        node = self.training_service.register_node(
            ComputeNodeRegistration(
                "fake-node",
                "local",
                "https://worker.example.test/agent",
                str(self.models_root),
                str(self.datasets_root),
                str(self.outputs_root),
                {"purpose": "stage-5"},
            )
        )
        self.node_id = node["id"]
        self.heartbeat(system_memory=100_000, gpu_memory=100_000, disk=100_000)
        self.text_job, self.text_adapter, self.text_checkpoint = self.train(
            self.text_model["id"], self.text_dataset, "qwen", qlora=False
        )
        self.image_job, self.image_adapter, self.image_checkpoint = self.train(
            self.image_model["id"], self.image_dataset, "qwen3_vl", qlora=True
        )
        self.result_service = ResultService(
            self.root / "service.sqlite3",
            self.result_root,
            model_service=self.model_service,
            dataset_service=self.dataset_service,
            training_service=self.training_service,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def heartbeat(self, *, system_memory, gpu_memory, disk):
        return self.training_service.heartbeat(
            self.node_id,
            [GpuReport(0, "Fake GPU", 100_000, gpu_memory, 0.0)],
            disk_free_bytes=disk,
            system_memory_free_bytes=system_memory,
            driver_version="fake-driver",
        )

    def train(self, model_id, dataset_id, template, *, qlora):
        parameters = {"stage": "sft", "finetuning_type": "lora", "template": template}
        if qlora:
            parameters.update({"quantization_bit": 4, "quantization_method": "bnb"})
        job = self.training_service.create_job(
            JobSubmission(
                "fake training",
                model_id,
                dataset_id,
                parameters,
                node_id=self.node_id,
            )
        )
        completed = FakeWorkerAgent(self.training_service, self.node_id).run_next()
        artifacts = self.training_service.list_artifacts(job["id"])
        adapter = next(item for item in artifacts if item["kind"] == "adapter")
        checkpoint = next(item for item in artifacts if item["kind"] == "checkpoint")
        assert completed["state"] == "SUCCEEDED"
        return completed, adapter, checkpoint

    def assert_result_error(self, code, callback):
        with self.assertRaises(ResultServiceError) as context:
            callback()
        assert context.exception.code == code

    def test_training_manifest_traces_config_data_logs_checkpoints_and_adapter(self):
        manifest = self.result_service.training_manifest(self.image_job["id"])
        assert manifest["complete"] is True
        assert manifest["model_id"] == self.image_model["id"]
        assert manifest["dataset_snapshot_id"] == self.image_dataset
        assert manifest["code_version"] == "4451765a6b04ff08a6c5650f5953513608ae9e64"
        assert manifest["config_snapshot"]["quantization_bit"] == 4
        assert {artifact["kind"] for artifact in manifest["artifacts"]} == {"adapter", "checkpoint", "log"}
        assert manifest["logs_url"].endswith("?download=true")
        assert manifest["training_curve"] == [
            {
                "sequence": manifest["training_curve"][0]["sequence"],
                "loss": 1.0,
                "learning_rate": 0.0001,
                "epoch": 1.0,
                "progress": 0.5,
                "gpu_memory_bytes": None,
            }
        ]
        assert self.image_adapter["metadata"]["adapter_type"] == "qlora"
        assert self.image_adapter["metadata"]["base_model_id"] == self.image_model["id"]

    def test_fake_chat_supports_base_adapter_text_and_image_deterministically(self):
        base = self.result_service.chat(
            model_id=self.text_model["id"],
            artifact_id=None,
            messages=[{"role": "user", "content": "Hello"}],
        )
        adapter = self.result_service.chat(
            model_id=self.text_model["id"],
            artifact_id=self.text_adapter["id"],
            messages=[{"role": "user", "content": "Hello"}],
        )
        repeated = self.result_service.chat(
            model_id=self.text_model["id"],
            artifact_id=self.text_adapter["id"],
            messages=[{"role": "user", "content": "Hello"}],
        )
        image = self.result_service.chat(
            model_id=self.image_model["id"],
            artifact_id=self.image_adapter["id"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "asset_id": "image-asset-fake"},
                        {"type": "text", "text": "Describe it"},
                    ],
                }
            ],
        )
        assert base["artifact_id"] is None
        assert adapter["message"] == repeated["message"]
        assert base["message"] != adapter["message"]
        assert image["modality"] == "image_text"
        assert image["usage"]["image_count"] == 1
        assert all(result["fake"] for result in (base, adapter, image))

    def test_chat_load_validation_never_silently_falls_back_to_base_model(self):
        self.assert_result_error(
            "ARTIFACT_NOT_FOUND",
            lambda: self.result_service.chat(
                model_id=self.text_model["id"],
                artifact_id="artifact-missing",
                messages=[{"role": "user", "content": "Hello"}],
            ),
        )
        self.assert_result_error(
            "ADAPTER_BASE_MODEL_MISMATCH",
            lambda: self.result_service.chat(
                model_id=self.image_model["id"],
                artifact_id=self.text_adapter["id"],
                messages=[{"role": "user", "content": "Hello"}],
            ),
        )
        self.assert_result_error(
            "MODEL_TEMPLATE_MISMATCH",
            lambda: self.result_service.chat(
                model_id=self.text_model["id"],
                artifact_id=None,
                template="wrong",
                messages=[{"role": "user", "content": "Hello"}],
            ),
        )
        self.assert_result_error(
            "TOKENIZER_NOT_AVAILABLE",
            lambda: self.result_service.chat(
                model_id=self.text_model["id"],
                artifact_id=None,
                tokenizer="broken",
                messages=[{"role": "user", "content": "Hello"}],
            ),
        )
        self.assert_result_error(
            "IMAGE_PROCESSOR_NOT_AVAILABLE",
            lambda: self.result_service.chat(
                model_id=self.image_model["id"],
                artifact_id=None,
                image_processor="broken",
                messages=[
                    {"role": "user", "content": [{"type": "image", "asset_id": "image-asset-fake"}]}
                ],
            ),
        )

    def test_evaluate_predict_outputs_deterministic_metrics_predictions_and_manifest(self):
        run = self.result_service.evaluate(
            model_id=self.image_model["id"],
            artifact_id=self.image_checkpoint["id"],
            dataset_snapshot_id=self.image_dataset,
        )
        assert run["state"] == "SUCCEEDED"
        assert run["metrics"] == {"eval_loss": 1.0, "exact_match": 0.5, "sample_count": 1}
        predictions = Path(run["prediction_path"]).read_text(encoding="utf-8")
        assert "fake-prediction-" in predictions
        assert json.loads(Path(run["manifest_path"]).read_text(encoding="utf-8"))["fake"] is True

        restarted = ResultService(
            self.root / "service.sqlite3",
            self.result_root,
            model_service=self.model_service,
            dataset_service=self.dataset_service,
            training_service=self.training_service,
        )
        assert restarted.get_evaluation(run["id"]) == run

    def test_fake_export_generates_merge_manifest_after_qlora_resource_preflight(self):
        exported = self.result_service.export_adapter(
            model_id=self.image_model["id"],
            adapter_id=self.image_adapter["id"],
            node_id=self.node_id,
            output_name="image-model-merged",
        )
        assert exported["state"] == "SUCCEEDED"
        assert exported["config"]["operation"] == "merge_adapter"
        assert exported["config"]["adapter_type"] == "qlora"
        assert exported["requirements"] == {
            "model_bytes": 4_000,
            "required_disk_bytes": 8_000,
            "required_gpu_memory_bytes": 4_000,
            "required_system_memory_bytes": 8_000,
        }
        marker = Path(exported["output_path"]) / "merged-model.fake.json"
        assert json.loads(marker.read_text(encoding="utf-8"))["warning"] == "No model weights were read or merged."
        assert Path(exported["manifest_path"]).is_file()

    def test_export_rejects_wrong_artifacts_paths_precision_and_insufficient_resources_before_start(self):
        self.assert_result_error(
            "ARTIFACT_KIND_NOT_SUPPORTED",
            lambda: self.result_service.export_adapter(
                model_id=self.text_model["id"],
                adapter_id=self.text_checkpoint["id"],
                node_id=self.node_id,
                output_name="bad-kind",
            ),
        )
        self.assert_result_error(
            "EXPORT_NAME_INVALID",
            lambda: self.result_service.export_adapter(
                model_id=self.text_model["id"],
                adapter_id=self.text_adapter["id"],
                node_id=self.node_id,
                output_name="../escape",
            ),
        )

        class QuantizedModelService:
            def get(inner_self, model_id):
                model = self.model_service.get(model_id)
                return {**model, "weight_precision": "int4"}

        quantized_result_service = ResultService(
            self.root / "service.sqlite3",
            self.result_root,
            model_service=QuantizedModelService(),
            dataset_service=self.dataset_service,
            training_service=self.training_service,
        )
        self.assert_result_error(
            "BASE_MODEL_PRECISION_UNSUPPORTED",
            lambda: quantized_result_service.export_adapter(
                model_id=self.image_model["id"],
                adapter_id=self.image_adapter["id"],
                node_id=self.node_id,
                output_name="quantized-base",
            ),
        )
        self.heartbeat(system_memory=1, gpu_memory=100_000, disk=100_000)
        self.assert_result_error(
            "SYSTEM_MEMORY_INSUFFICIENT",
            lambda: self.result_service.export_adapter(
                model_id=self.image_model["id"],
                adapter_id=self.image_adapter["id"],
                node_id=self.node_id,
                output_name="not-created",
            ),
        )
        assert not any(self.outputs_root.rglob("not-created"))


if __name__ == "__main__":
    unittest.main()
