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
from unittest.mock import patch

from PIL import Image

from sft_platform.extras.mvp_policy import FakeExecutor
from sft_platform.mvp.assets import (
    MAX_IMAGE_BYTES,
    MAX_UPLOAD_FILE_BYTES,
    AssetValidationError,
    DatasetAssetService,
    FakeHub,
    ModelAssetService,
    ModelReference,
    build_upstream_model_catalog,
)


class MvpModelAssetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.fake_hub = FakeHub(
            {
                ("huggingface", "company/text-model"): {
                    "revision": "fake-hf-revision",
                    "cache_state": "cached",
                    "private": True,
                    "parameter_count": 1_000_000,
                    "template": "qwen",
                    "context_length": 8192,
                    "modalities": ["text"],
                },
                ("modelscope", "company/image-model"): {
                    "revision": "fake-ms-revision",
                    "cache_state": "not_cached",
                    "parameter_count": 2_000_000,
                    "template": "qwen3_vl",
                    "context_length": 32768,
                    "modalities": ["text", "image"],
                },
            }
        )
        self.service = ModelAssetService([self.models], self.fake_hub)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_local_huggingface_and_modelscope_references_use_metadata_only(self):
        local_model = self.models / "local-model"
        local_model.mkdir()
        (local_model / "config.json").write_text(
            json.dumps(
                {
                    "_commit_hash": "fake-local-revision",
                    "parameter_count": 3_000_000,
                    "template": "qwen3_vl",
                    "max_position_embeddings": 16384,
                    "modalities": ["text", "image"],
                }
            ),
            encoding="utf-8",
        )
        local = self.service.register(ModelReference("local", "local", str(local_model)))
        huggingface = self.service.register(
            ModelReference("hf", "huggingface", "company/text-model", credential_id="cred-hf-private")
        )
        modelscope = self.service.register(
            ModelReference("ms", "modelscope", "company/image-model", credential_id="cred-ms-private")
        )

        assert local["compatibility"]["modes"] == ["text_sft", "image_text_sft"]
        assert huggingface["cache_state"] == "cached"
        assert modelscope["compatibility"]["mvp_compatible"] is True
        assert "credential_id" not in huggingface
        assert "credential_id" not in modelscope
        assert len(self.service.list()) == 3

    def test_model_path_repository_credentials_and_compatibility_failures_are_explicit(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "config.json").write_text("{}", encoding="utf-8")
        checks = (
            (
                ModelReference("outside", "local", str(outside)),
                "MODEL_PATH_OUTSIDE_ROOT",
            ),
            (
                ModelReference("url", "huggingface", "https://example.test/model"),
                "MODEL_REPOSITORY_INVALID",
            ),
            (
                ModelReference("private", "huggingface", "company/text-model"),
                "MODEL_CREDENTIAL_REQUIRED",
            ),
            (
                ModelReference("secret", "huggingface", "company/text-model", credential_id="hf_plain_token"),
                "PLAINTEXT_CREDENTIAL_NOT_ALLOWED",
            ),
        )
        for reference, code in checks:
            with self.subTest(code=code), self.assertRaises(AssetValidationError) as context:
                self.service.validate(reference)
            assert context.exception.code == code

        self.fake_hub.add("huggingface", "company/audio-model", {"modalities": ["audio"]})
        incompatible = self.service.validate(ModelReference("audio", "huggingface", "company/audio-model"))
        assert incompatible["compatibility"]["mvp_compatible"] is False
        assert incompatible["compatibility"]["modes"] == []

    def test_model_registry_can_persist_without_credentials(self):
        state_root = self.root / "state"
        registered = ModelAssetService([self.models], self.fake_hub, state_root=state_root).register(
            ModelReference("hf", "huggingface", "company/text-model", credential_id="cred-hf-private")
        )
        restarted = ModelAssetService([self.models], self.fake_hub, state_root=state_root)
        assert restarted.get(registered["id"])["id"] == registered["id"]
        assert "cred-hf-private" not in (state_root / "models.json").read_text(encoding="utf-8")

    def test_upstream_registry_is_preserved_and_marked(self):
        catalog = build_upstream_model_catalog()
        # The 639 upstream registrations contain one duplicate name; the effective registry has 638 unique choices.
        assert len(catalog) == 638
        assert sum(item["modality"] == "image_text" for item in catalog) == 176
        assert any(not item["mvp_compatible"] for item in catalog)
        assert all(item["sources"] for item in catalog)


class MvpDatasetAssetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.datasets = self.root / "datasets"
        self.media = self.root / "media"
        self.datasets.mkdir()
        self.media.mkdir()
        Image.new("RGB", (32, 24), "red").save(self.media / "sample.jpg", format="JPEG")
        Image.new("RGB", (16, 12), "blue").save(self.media / "sample.webp", format="WEBP")
        Image.new("RGB", (8, 6), "green").save(self.media / "sample.png", format="PNG")
        self.service = DatasetAssetService([self.datasets], [self.media])

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_json(self, name, records):
        path = self.datasets / name
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return path

    def write_jsonl(self, name, records):
        path = self.datasets / name
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_alpaca_field_mapping_preview_statistics_and_snapshot_match(self):
        path = self.write_json(
            "alpaca.json",
            [
                {
                    "prompt_text": "Describe <image>",
                    "extra_input": "briefly",
                    "answer_text": "A red rectangle.",
                    "pictures": ["sample.jpg"],
                },
                {
                    "prompt_text": "Say hello",
                    "extra_input": "",
                    "answer_text": "Hello",
                    "pictures": [],
                },
            ],
        )
        inspection = self.service.inspect(
            path,
            "alpaca",
            mapping={
                "instruction": "prompt_text",
                "input": "extra_input",
                "output": "answer_text",
                "images": "pictures",
            },
        )
        assert inspection.ready is True
        assert inspection.statistics["sample_count"] == 2
        assert inspection.statistics["anomaly_count"] == 0
        assert inspection.statistics["image_count"] == 1
        assert inspection.preview == self.service.get_snapshot(inspection.snapshot_id)
        assert inspection.preview[0]["images"] == [str((self.media / "sample.jpg").resolve())]
        fake_result = FakeExecutor().execute(
            {
                "stage": "sft",
                "do_train": True,
                "model_name_or_path": "fixtures/models/image-text-sft",
                "dataset": "mapped-alpaca",
                "finetuning_type": "lora",
                "output_dir": "fixtures/outputs/mapped-alpaca",
            },
            dataset_snapshot_id=inspection.snapshot_id,
        )
        assert fake_result["dataset_snapshot_id"] == inspection.snapshot_id

        previous_snapshot = inspection.snapshot_id
        Image.new("RGB", (33, 24), "red").save(self.media / "sample.jpg", format="JPEG")
        changed = self.service.inspect(path, "alpaca", mapping={
            "instruction": "prompt_text",
            "input": "extra_input",
            "output": "answer_text",
            "images": "pictures",
        })
        assert changed.snapshot_id != previous_snapshot

    def test_sharegpt_jsonl_and_openai_multimodal_content_convert_to_one_contract(self):
        sharegpt = self.write_jsonl(
            "sharegpt.jsonl",
            [
                {
                    "chat": [
                        {"speaker": "human", "text": "What is shown? <image>"},
                        {"speaker": "gpt", "text": "A green image."},
                    ],
                    "pictures": "sample.png",
                }
            ],
        )
        sharegpt_result = self.service.inspect(
            sharegpt,
            "sharegpt",
            mapping={"messages": "chat", "role": "speaker", "content": "text", "images": "pictures"},
        )
        assert sharegpt_result.ready is True
        assert sharegpt_result.preview[0]["messages"][0]["role"] == "user"

        openai = self.write_json(
            "openai.json",
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "sample.webp"}},
                                {"type": "text", "text": "Describe it."},
                            ],
                        },
                        {"role": "assistant", "content": "A blue rectangle."},
                    ]
                }
            ],
        )
        openai_result = self.service.inspect(openai, "openai")
        assert openai_result.ready is True
        assert openai_result.preview[0]["messages"][0]["content"] == "<image>Describe it."
        assert openai_result.statistics["image_width"]["max"] == 16

    def test_bad_images_placeholder_mismatch_and_unsupported_media_locate_samples(self):
        (self.media / "bad.png").write_bytes(b"not-an-image")
        Image.new("RGB", (4097, 1), "black").save(self.media / "wide.png", format="PNG")
        path = self.write_json(
            "invalid.json",
            [
                {"instruction": "Missing <image>", "input": "", "output": "x", "images": ["missing.png"]},
                {"instruction": "Bad <image>", "input": "", "output": "x", "images": ["bad.png"]},
                {"instruction": "Wide <image>", "input": "", "output": "x", "images": ["wide.png"]},
                {"instruction": "Mismatch", "input": "", "output": "x", "images": ["sample.png"]},
                {"instruction": "Video", "input": "", "output": "x", "videos": ["sample.mp4"]},
            ],
        )
        result = self.service.inspect(path, "alpaca")
        assert result.ready is False
        assert result.snapshot_id is None
        assert result.statistics["sample_count"] == 5
        assert result.statistics["anomaly_count"] == 5
        assert [error["sample_index"] for error in result.errors] == [0, 1, 2, 3, 4]
        assert {error["code"] for error in result.errors} == {
            "IMAGE_NOT_FOUND",
            "IMAGE_INVALID",
            "IMAGE_DIMENSION_TOO_LARGE",
            "IMAGE_PLACEHOLDER_MISMATCH",
            "DATASET_MODALITY_NOT_SUPPORTED",
        }

    def test_upload_dataset_total_image_size_and_image_format_limits_are_enforced(self):
        tiny_dataset = self.write_json(
            "tiny.json",
            [{"instruction": "x", "input": "", "output": "y"}],
        )
        with patch("sft_platform.mvp.assets.MAX_DATASET_BYTES", 1):
            with self.assertRaises(AssetValidationError) as context:
                self.service.inspect(tiny_dataset, "alpaca")
        assert context.exception.code == "DATASET_TOO_LARGE"

        oversized_dataset = self.datasets / "oversized.json"
        with oversized_dataset.open("wb") as file:
            file.truncate(MAX_UPLOAD_FILE_BYTES + 1)
        with self.assertRaises(AssetValidationError) as context:
            self.service.inspect(oversized_dataset, "alpaca")
        assert context.exception.code == "DATASET_FILE_TOO_LARGE"

        oversized_image = self.media / "oversized.png"
        with oversized_image.open("wb") as file:
            file.truncate(MAX_IMAGE_BYTES + 1)
        Image.new("RGB", (10, 10), "yellow").save(self.media / "sample.gif", format="GIF")
        path = self.write_json(
            "image-limits.json",
            [
                {
                    "instruction": "Large <image>",
                    "input": "",
                    "output": "x",
                    "images": ["oversized.png"],
                },
                {
                    "instruction": "GIF <image>",
                    "input": "",
                    "output": "x",
                    "images": ["sample.gif"],
                },
            ],
        )
        result = self.service.inspect(path, "alpaca")
        assert [error["code"] for error in result.errors] == ["IMAGE_TOO_LARGE", "IMAGE_FORMAT_NOT_SUPPORTED"]

    def test_upload_index_and_normalized_snapshot_survive_service_restart(self):
        state_root = self.root / "state"
        service = DatasetAssetService([self.datasets], [self.media], state_root=state_root)
        payload = json.dumps([{"instruction": "Hello", "input": "", "output": "World"}]).encode()
        with tempfile.TemporaryFile() as stream:
            stream.write(payload)
            stream.seek(0)
            uploaded = service.store_upload("persistent.json", stream)
        inspected = service.inspect_upload(uploaded["id"], "alpaca")

        restarted = DatasetAssetService([self.datasets], [self.media], state_root=state_root)
        assert restarted.inspect_upload(uploaded["id"], "alpaca").snapshot_id == inspected.snapshot_id
        assert restarted.get_snapshot(inspected.snapshot_id) == inspected.preview

    def test_path_file_type_message_and_remote_image_rejections_are_stable(self):
        outside = self.root / "outside.json"
        outside.write_text("[]", encoding="utf-8")
        csv_path = self.datasets / "data.csv"
        csv_path.write_text("instruction,output", encoding="utf-8")
        invalid_messages = self.write_json(
            "invalid-messages.json",
            [{"messages": [{"role": "assistant", "content": "wrong order"}]}],
        )
        remote_image = self.write_json(
            "remote-image.json",
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": "https://example.test/image.png"}],
                        },
                        {"role": "assistant", "content": "x"},
                    ]
                }
            ],
        )

        for path, formatting, code in (
            (outside, "alpaca", "DATASET_PATH_OUTSIDE_ROOT"),
            (csv_path, "alpaca", "DATASET_FILE_TYPE_NOT_SUPPORTED"),
        ):
            with self.subTest(code=code), self.assertRaises(AssetValidationError) as context:
                self.service.inspect(path, formatting)
            assert context.exception.code == code

        messages_result = self.service.inspect(invalid_messages, "openai")
        assert messages_result.errors[0]["code"] == "MESSAGE_COUNT_INVALID"
        remote_result = self.service.inspect(remote_image, "openai")
        assert remote_result.errors[0]["code"] == "IMAGE_REFERENCE_INVALID"


if __name__ == "__main__":
    unittest.main()
