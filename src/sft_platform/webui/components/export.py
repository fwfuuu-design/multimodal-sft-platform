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

"""Adapter-merge-only Export tab."""

from collections.abc import Generator
from typing import TYPE_CHECKING

from ...extras.misc import torch_gc
from ...extras.packages import is_gradio_available
from ...train.tuner import export_model
from ..common import get_save_dir, load_config
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: str | list[str],
    template: str,
    export_size: int,
    export_dir: str,
) -> Generator[str, None, None]:
    user_config = load_config()
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif finetuning_type != "lora" or not isinstance(checkpoint_path, list) or len(checkpoint_path) != 1:
        error = ALERTS["err_no_adapter"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    args = {
        "model_name_or_path": model_path,
        "adapter_name_or_path": get_save_dir(model_name, finetuning_type, checkpoint_path[0]),
        "cache_dir": user_config.get("cache_dir"),
        "finetuning_type": "lora",
        "stage": "sft",
        "template": template,
        "export_dir": export_dir,
        "export_size": export_size,
        "export_device": "cpu",
        "trust_remote_code": True,
    }
    yield ALERTS["info_exporting"][lang]
    export_model(args)
    torch_gc()
    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> dict[str, "Component"]:
    gr.Markdown("选择顶部区域中的单个 Adapter，将其与基础模型合并为可独立加载的模型目录。")
    with gr.Row():
        export_dir = gr.Textbox(scale=3)
        export_size = gr.Slider(minimum=1, maximum=100, value=5, step=1, scale=1)

    export_btn = gr.Button(variant="primary")
    info_box = gr.Textbox(show_label=False, interactive=False)
    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            export_size,
            export_dir,
        ],
        [info_box],
    )
    return {"export_size": export_size, "export_dir": export_dir, "export_btn": export_btn, "info_box": info_box}
