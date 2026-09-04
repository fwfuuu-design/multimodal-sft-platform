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

import ast
import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "sft_platform"


def read_source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def sha256(relative_path: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()


def literal_assignment(tree: ast.AST, name: str):
    for node in tree.body:
        is_target = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        )
        if is_target:
            return ast.literal_eval(node.value)

    raise AssertionError(f"Assignment not found: {name}")


class MvpSurfaceBaselineTest(unittest.TestCase):
    """Freeze upstream surfaces that later MVP pruning must preserve or review."""

    def test_four_product_tabs_and_cli_entrypoints_exist(self):
        interface = read_source("src/sft_platform/webui/interface.py")
        launcher = read_source("src/sft_platform/launcher.py")

        for tab in ("训练", "评估与预测", "对话验证", "合并导出"):
            assert f'gr.Tab("{tab}")' in interface

        for command in ("api", "bootstrap", "backup", "restore", "train", "webui"):
            assert f'command == "{command}"' in launcher
        for command in ("chat", "eval", "export", "webchat", "mvp-api"):
            assert f'command == "{command}"' not in launcher

    def test_current_api_and_webui_execution_boundary_is_explicit(self):
        runtime = read_source("src/sft_platform/api/mvp_runtime.py")
        runner = read_source("src/sft_platform/webui/runner.py")
        chatter = read_source("src/sft_platform/webui/chatter.py")

        assert "create_secure_mvp_app" in runtime
        assert not (REPO_ROOT / "src/sft_platform/api/app.py").exists()
        assert "Popen(" in runner
        assert '["sft-train", "train", save_cmd(args)]' in runner
        assert "from ..chat import ChatModel" in chatter

    def test_product_training_choices_are_physically_pruned(self):
        tree = ast.parse(read_source("src/sft_platform/extras/constants.py"))

        assert literal_assignment(tree, "TRAINING_STAGES") == {"Supervised Fine-Tuning": "sft"}
        assert literal_assignment(tree, "PEFT_METHODS") == {"lora"}

    def test_product_sft_case_snapshots_cover_text_image_lora_and_qlora(self):
        cases = json.loads(read_source("tests/fixtures/mvp_sft_cases.json"))
        assert set(cases) == {"text_lora", "text_qlora", "image_text_lora", "image_text_qlora"}

        for name, config in cases.items():
            assert config["stage"] == "sft", name
            assert config["do_train"] is True, name
            assert config["finetuning_type"] == "lora", name
            assert config["model_name_or_path"].startswith("fixtures/models/"), name
            assert config["output_dir"].startswith("fixtures/outputs/"), name

            is_qlora = name.endswith("_qlora")
            assert config.get("quantization_bit") == (4 if is_qlora else None), name
            if is_qlora:
                assert config["quantization_method"] == "bnb", name

            is_multimodal = name.startswith("image_text_")
            assert (config["dataset"] == "image_text_sft") == is_multimodal, name

    def test_model_registry_names_match_locked_upstream(self):
        tree = ast.parse(read_source("src/sft_platform/extras/constants.py"))
        groups = []
        multimodal = []

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "register_model_group":
                continue

            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            model_dict = keywords.get("models")
            if not isinstance(model_dict, ast.Dict):
                continue

            names = [ast.literal_eval(key) for key in model_dict.keys]
            groups.append(names)
            marker = keywords.get("multimodal")
            if isinstance(marker, ast.Constant) and marker.value is True:
                multimodal.extend(names)

        registry_payload = "\n".join("\0".join(group) for group in groups).encode()
        assert len(groups) == 133
        assert sum(map(len, groups)) == 639
        assert len(multimodal) == 176
        assert (
            hashlib.sha256(registry_payload).hexdigest()
            == "e92ba1102c8acc5bf356ae7332fbcd4676fe05b743e39459c4f453a8d770596f"
        )

    def test_shared_directories_required_by_product_flows_exist(self):
        for relative_path in ("api", "chat", "data", "extras", "model", "train/sft", "webui"):
            assert (SOURCE_ROOT / relative_path).is_dir(), relative_path

    def test_cuda_npu_and_rocm_adapter_sources_are_retained(self):
        required = {
            "src/sft_platform/model/kernels/ops/mlp/cuda_fused_moe.py": "CudaFusedMoEKernel",
            "src/sft_platform/model/kernels/ops/mlp/npu_fused_moe.py": "NpuFusedMoEKernel",
            "src/sft_platform/model/kernels/ops/mlp/npu_swiglu.py": "NpuSwiGluKernel",
            "src/sft_platform/model/kernels/ops/rms_norm/npu_rms_norm.py": "NpuRMSNormKernel",
            "src/sft_platform/model/kernels/ops/rope/npu_rope.py": "NpuRoPEKernel",
            "src/sft_platform/third_party/triton/utils.py": "get_available_device",
        }
        for path, marker in required.items():
            assert marker in (REPO_ROOT / path).read_text(encoding="utf-8"), path
        assert (REPO_ROOT / "docker/docker-npu/Dockerfile").is_file()
        assert (REPO_ROOT / "docker/docker-rocm/Dockerfile").is_file()


if __name__ == "__main__":
    unittest.main()
