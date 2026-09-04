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
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODIFICATION_NOTICE = "Modified in 2026 for the Multimodal Fine-tuning Platform MVP."
MODIFIED_PYTHON_FILES = (
    "src/sft_platform/api/assets.py",
    "src/sft_platform/api/jobs.py",
    "src/sft_platform/api/mvp_runtime.py",
    "src/sft_platform/api/results.py",
    "src/sft_platform/api/security.py",
    "src/sft_platform/cli.py",
    "src/sft_platform/data/loader.py",
    "src/sft_platform/data/processor/__init__.py",
    "src/sft_platform/data/processor/generation.py",
    "src/sft_platform/data/processor/supervised.py",
    "src/sft_platform/extras/constants.py",
    "src/sft_platform/extras/mvp_policy.py",
    "src/sft_platform/hparams/__init__.py",
    "src/sft_platform/hparams/finetuning_args.py",
    "src/sft_platform/hparams/parser.py",
    "src/sft_platform/launcher.py",
    "src/sft_platform/mvp/__init__.py",
    "src/sft_platform/mvp/assets.py",
    "src/sft_platform/mvp/jobs.py",
    "src/sft_platform/mvp/operations.py",
    "src/sft_platform/mvp/results.py",
    "src/sft_platform/model/adapter.py",
    "src/sft_platform/mvp/security.py",
    "src/sft_platform/train/tuner.py",
    "src/sft_platform/webui/chatter.py",
    "src/sft_platform/webui/common.py",
    "src/sft_platform/webui/components/chatbot.py",
    "src/sft_platform/webui/components/export.py",
    "src/sft_platform/webui/components/infer.py",
    "src/sft_platform/webui/components/top.py",
    "src/sft_platform/webui/components/train.py",
    "src/sft_platform/webui/control.py",
    "src/sft_platform/webui/engine.py",
    "src/sft_platform/webui/manager.py",
    "src/sft_platform/webui/runner.py",
    "tests/test_mvp_asset_api.py",
    "tests/test_mvp_assets.py",
    "tests/test_mvp_job_api.py",
    "tests/test_mvp_jobs.py",
    "tests/test_mvp_operations.py",
    "tests/test_mvp_release_gate.py",
    "tests/test_mvp_result_api.py",
    "tests/test_mvp_results.py",
    "tests/test_mvp_runtime.py",
    "tests/test_mvp_security.py",
    "tests/test_mvp_surface_baseline.py",
    "tests/test_mvp_training_policy.py",
)


def test_license_attribution_and_modification_notices_are_present():
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("Apache License\n")
    assert "Version 2.0, January 2004" in license_text
    assert MODIFICATION_NOTICE in (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for relative_path in MODIFIED_PYTHON_FILES:
        assert MODIFICATION_NOTICE in (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"), relative_path


def test_versioned_openapi_contract_is_present():
    contract = REPOSITORY_ROOT / "docs" / "openapi-v1.json"
    assert contract.is_file()
    schema = json.loads(contract.read_text(encoding="utf-8"))
    assert schema["info"]["version"] == "1.0.0"
    assert schema["security"] == [{"OIDCBearer": []}]
    assert len(schema["paths"]) == 38


def test_excluded_training_modules_are_physically_removed_after_decoupling():
    for directory in ("pt", "rm", "ppo", "dpo", "kto", "hyper_parallel", "mca", "megatron_bridge"):
        assert not (REPOSITORY_ROOT / "src" / "sft_platform" / "train" / directory).exists(), directory
    for path in (
        "src/sft_platform/eval",
        "src/sft_platform/api/app.py",
        "src/sft_platform/api/chat.py",
        "src/sft_platform/api/common.py",
        "src/sft_platform/api/protocol.py",
        "src/sft_platform/data/processor/pretrain.py",
        "src/sft_platform/data/processor/pairwise.py",
        "src/sft_platform/data/processor/feedback.py",
    ):
        assert not (REPOSITORY_ROOT / path).exists(), path

    tuner = (REPOSITORY_ROOT / "src" / "sft_platform" / "train" / "tuner.py").read_text(encoding="utf-8")
    assert "from .sft import run_sft" in tuner
    for import_name in ("run_pt", "run_rm", "run_ppo", "run_dpo", "run_kto", "use_ray", "Ollama"):
        assert import_name not in tuner


def test_second_pruning_batch_removed_alternate_frameworks_and_upstream_product_surface():
    assert REPOSITORY_ROOT.name == "SFTPlatform"
    for path in (
        "src/sft_platform/v1",
        "tests_v1",
        "src/sft_platform/chat/vllm_engine.py",
        "src/sft_platform/chat/sglang_engine.py",
        "src/sft_platform/model/model_utils/valuehead.py",
        "src/sft_platform/model/model_utils/unsloth.py",
        "src/sft_platform/train/fp8_utils.py",
        "scripts",
        "assets",
    ):
        assert not (REPOSITORY_ROOT / path).exists(), path

    assert (REPOSITORY_ROOT / "src/sft_platform/model/kernels/interface.py").is_file()
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "multimodal-sft-platform"' in pyproject
    assert "sft_platform-cli" not in pyproject
    assert 'lmf = ' not in pyproject
    for excluded_dependency in ("trl", "torchaudio", "torchdata", "sse-starlette", "tyro"):
        assert f'"{excluded_dependency}' not in pyproject

    upstream_brand = re.compile(r"llama[ -]?" + r"factory|llama" + r"factory", re.IGNORECASE)
    for root in ("README.md", "README_zh.md", "examples", "data", "docker"):
        path = REPOSITORY_ROOT / root
        files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for file in files:
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".ico"}:
                continue
            assert not upstream_brand.search(file.read_text(encoding="utf-8")), file


def test_formal_runtime_is_explicit_and_fail_closed():
    launcher = (REPOSITORY_ROOT / "src" / "sft_platform" / "launcher.py").read_text(encoding="utf-8")
    runtime = (REPOSITORY_ROOT / "src" / "sft_platform" / "api" / "mvp_runtime.py").read_text(encoding="utf-8")
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'sft-train = "sft_platform.cli:main"' in pyproject
    for command in ("api", "bootstrap", "backup", "restore"):
        assert f'command == "{command}"' in launcher
    assert "from .api.app import run_api" not in launcher
    assert "sft-train api" in launcher
    assert "create_secure_mvp_app" in runtime
    assert "MVP_TOKEN_VERIFIER_FACTORY" in runtime
    assert "MVP_HUB_INSPECTOR_FACTORY" in runtime
    assert "MVP_MODEL_ADAPTER_FACTORY" in runtime
    assert "or FakeOIDCVerifier" not in runtime
    assert "or FakeModelAdapter" not in runtime
