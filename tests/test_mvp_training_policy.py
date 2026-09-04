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

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "src" / "sft_platform" / "extras" / "mvp_policy.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mvp_sft_cases.json"


def load_policy_module():
    spec = importlib.util.spec_from_file_location("mvp_policy_under_test", POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = load_policy_module()


class MvpTrainingPolicyTest(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def assert_policy_error(self, code, config=None, environ=None):
        with self.assertRaises(policy.MvpPolicyError) as context:
            if environ is not None:
                policy.validate_mvp_runtime_environment(environ)
            else:
                policy.validate_mvp_train_config(config)
        assert context.exception.code == code
        return context.exception

    def test_four_sft_lora_qlora_cases_generate_controlled_dry_runs(self):
        for name, config in self.cases.items():
            with self.subTest(name=name):
                report = policy.build_mvp_dry_run(config)
                parsed_yaml = json.loads(report["yaml"])
                assert parsed_yaml["stage"] == "sft"
                assert parsed_yaml["finetuning_type"] == "lora"
                assert report["mode"] == ("qlora" if config.get("quantization_bit") == 4 else "lora")
                assert report["argv"] == ["sft-train", "train", "<controlled-job.yaml>"]
                assert report["execution"]["state"] == "SUCCEEDED"
                assert all(artifact["fake"] for artifact in report["execution"]["artifacts"])

    def test_cli_dry_run_reads_a_controlled_config_before_training_imports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "job.json"
            config_path.write_text(json.dumps(self.cases["text_lora"]), encoding="utf-8")
            request = policy.read_mvp_train_request([str(config_path), "learning_rate=0.0002"])
            assert request["learning_rate"] == "0.0002"
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report = policy.run_mvp_dry_run([str(config_path)])
            assert report["execution"]["state"] == "SUCCEEDED"
            assert '"dry_run": true' in output.getvalue()

    def test_excluded_stages_methods_and_quantization_are_rejected(self):
        base = self.cases["text_lora"]
        for stage in ("pt", "rm", "ppo", "dpo", "kto"):
            self.assert_policy_error("MVP_TRAINING_STAGE_NOT_SUPPORTED", {**base, "stage": stage})
        for method in ("full", "freeze", "oft"):
            self.assert_policy_error(
                "MVP_FINETUNING_METHOD_NOT_SUPPORTED", {**base, "finetuning_type": method}
            )
        self.assert_policy_error("MVP_QUANTIZATION_NOT_SUPPORTED", {**base, "quantization_bit": 8})
        self.assert_policy_error(
            "MVP_QUANTIZATION_BACKEND_NOT_SUPPORTED",
            {**base, "quantization_bit": 4, "quantization_method": "gptq"},
        )

    def test_advanced_algorithms_and_distributed_backends_are_rejected(self):
        base = self.cases["text_lora"]
        advanced_fields = (
            "enable_liger_kernel",
            "fp8",
            "pissa_init",
            "pissa_convert",
            "shift_attn",
            "train_from_scratch",
            "use_adam_mini",
            "use_apollo",
            "use_asft_loss",
            "use_badam",
            "use_dft_loss",
            "use_dora",
            "use_eaft_loss",
            "use_galore",
            "use_llama_pro",
            "use_muon",
            "use_rslora",
            "use_unsloth",
            "use_unsloth_gc",
        )
        for field in advanced_fields:
            with self.subTest(field=field):
                error = self.assert_policy_error(
                    "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED", {**base, field: True}
                )
                assert error.field == field

        for field, value in (
            ("deepspeed", "ds.json"),
            ("fp8_enable_fsdp_float8_all_gather", True),
            ("fsdp", "full_shard"),
            ("use_hyper_parallel", True),
            ("ray_num_workers", 2),
        ):
            with self.subTest(field=field):
                self.assert_policy_error(
                    "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED", {**base, field: value}
                )

    def test_runtime_allows_single_machine_and_rejects_multi_node_or_elastic(self):
        policy.validate_mvp_runtime_environment({"NNODES": "1", "NPROC_PER_NODE": "4"})
        self.assert_policy_error("MVP_MULTI_NODE_NOT_SUPPORTED", environ={"NNODES": "2"})
        self.assert_policy_error(
            "MVP_MULTI_NODE_NOT_SUPPORTED", environ={"NNODES": "1", "NODE_RANK": "1"}
        )
        self.assert_policy_error(
            "MVP_ELASTIC_TRAINING_NOT_SUPPORTED", environ={"NNODES": "1", "RDZV_ID": "job-1"}
        )
        self.assert_policy_error(
            "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED", environ={"NNODES": "1", "USE_RAY": "1"}
        )

    def test_text_and_image_are_allowed_but_audio_and_video_are_rejected(self):
        policy.reject_unsupported_training_media([], [])
        self.assert_policy_error(
            "MVP_VIDEO_TRAINING_NOT_SUPPORTED",
            {**self.cases["image_text_lora"], "use_audio_in_video": True},
        )
        self.assert_policy_error_for_media("MVP_VIDEO_TRAINING_NOT_SUPPORTED", videos=["fake.mp4"], audios=[])
        self.assert_policy_error_for_media("MVP_AUDIO_TRAINING_NOT_SUPPORTED", videos=[], audios=["fake.wav"])

    def assert_policy_error_for_media(self, code, videos, audios):
        with self.assertRaises(policy.MvpPolicyError) as context:
            policy.reject_unsupported_training_media(videos, audios)
        assert context.exception.code == code

    def test_fake_nodes_cover_local_cloud_single_and_multi_gpu_and_resume(self):
        config = self.cases["image_text_qlora"]
        for node_type in ("local", "cloud"):
            for gpu_count in (1, 2):
                with self.subTest(node_type=node_type, gpu_count=gpu_count):
                    executor = policy.FakeExecutor(node_type=node_type, gpu_count=gpu_count)
                    result = executor.execute(config, gpu_ids=list(range(gpu_count)))
                    assert result["node"]["type"] == node_type
                    assert result["node"]["allocated_gpu_ids"] == list(range(gpu_count))
                    metric_event = next(event for event in result["events"] if "metrics" in event)
                    assert set(metric_event["metrics"]) == {"loss", "learning_rate", "epoch", "progress", "gpu_memory_bytes"}
                    resumed = executor.resume(config, result["checkpoint"], gpu_ids=list(range(gpu_count)))
                    assert resumed["attempt"] == 2
                    assert resumed["state"] == "SUCCEEDED"

    def test_formal_entries_and_sft_processor_use_the_policy(self):
        expected_references = {
            "src/sft_platform/cli.py": "validate_mvp_runtime_environment",
            "src/sft_platform/launcher.py": "run_mvp_dry_run",
            "src/sft_platform/hparams/parser.py": "validate_mvp_train_config(raw_args)",
            "src/sft_platform/train/tuner.py": "validate_mvp_train_config(args)",
            "src/sft_platform/data/processor/supervised.py": "reject_unsupported_training_media(videos, audios)",
        }
        for relative_path, reference in expected_references.items():
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            assert reference in source, relative_path


if __name__ == "__main__":
    unittest.main()
