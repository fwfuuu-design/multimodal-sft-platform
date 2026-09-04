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
#
# Modified in 2026 for the multimodal fine-tuning product WebUI.

from pathlib import Path

from sft_platform.webui.common import gen_cmd
from sft_platform.webui.css import CSS
from sft_platform.webui.interface import create_ui
from sft_platform.webui.locales import LOCALES


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _choices(component: dict) -> list[str]:
    return [choice[1] for choice in component.get("props", {}).get("choices") or []]


def test_product_shell_and_four_original_workflows_are_present():
    config = create_ui().get_config_file()

    assert config["title"].startswith("多模态微调平台")
    assert [
        component["props"]["label"] for component in config["components"] if component["type"] == "tabitem"
    ] == ["训练", "评估与预测", "对话验证", "合并导出"]
    assert "--mvp-canvas: #f5f5f5" in CSS
    assert "--mvp-ink: #0a0a0a" in CSS


def test_visible_product_choices_exclude_non_mvp_capabilities():
    components = create_ui().get_config_file()["components"]
    visible_dropdown_choices = [
        _choices(component)
        for component in components
        if component["type"] == "dropdown" and component.get("props", {}).get("visible", True)
    ]

    assert ["zh", "en"] in visible_dropdown_choices
    assert ["lora"] in visible_dropdown_choices
    assert ["none", "4"] in visible_dropdown_choices
    assert ["Supervised Fine-Tuning"] in visible_dropdown_choices

    flattened = {choice for choices in visible_dropdown_choices for choice in choices}
    for excluded in ("full", "freeze", "oft", "rm", "ppo", "dpo", "kto", "pt", "vllm", "sglang", "openmind"):
        assert excluded not in flattened

    model_choices = next(choices for choices in visible_dropdown_choices if "Custom" in choices)
    assert all(not any(token in name.lower() for token in ("audio", "omni", "video")) for name in model_choices)


def test_audio_video_inputs_are_removed_and_public_command_is_used():
    components = create_ui().get_config_file()["components"]
    media = [component for component in components if component["type"] in {"audio", "video"}]

    assert media == []
    assert gen_cmd({"stage": "sft", "finetuning_type": "lora"}).startswith("```bash\nsft-train train ")


def test_excluded_controls_are_removed_instead_of_only_hidden():
    train_source = (REPOSITORY_ROOT / "src/sft_platform/webui/components/train.py").read_text(encoding="utf-8")
    export_source = (REPOSITORY_ROOT / "src/sft_platform/webui/components/export.py").read_text(encoding="utf-8")
    chat_source = (REPOSITORY_ROOT / "src/sft_platform/webui/components/chatbot.py").read_text(encoding="utf-8")
    for excluded in (
        "use_galore",
        "use_apollo",
        "use_badam",
        "use_swanlab",
        "use_dora",
        "use_pissa",
        "reward_model",
        "ds_stage",
        "video_max_pixels",
    ):
        assert excluded not in train_source
    for excluded in ("export_quantization_bit", "export_hub_model_id", "extra_args"):
        assert excluded not in export_source
    assert "gr.Video" not in chat_source
    assert "gr.Audio" not in chat_source


def test_default_brand_copy_has_no_upstream_promotion():
    for lang in ("zh", "en"):
        copy = LOCALES["title"][lang]["value"] + LOCALES["subtitle"][lang]["value"]
        assert ("LLaMA" + " Factory") not in copy
        assert "github.com/hiyouga" not in copy
        assert "sft_platform.net" not in copy
