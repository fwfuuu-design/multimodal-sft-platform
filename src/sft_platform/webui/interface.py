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

import os
import platform
import warnings

from ..extras.misc import fix_proxy, is_env_enabled
from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_eval_tab,
    create_export_tab,
    create_footer,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine import Engine


if is_gradio_available():
    import gradio as gr


def create_ui() -> "gr.Blocks":
    engine = Engine()
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'css' parameter in the Blocks constructor.*", category=DeprecationWarning)
        blocks = gr.Blocks(title=f"多模态微调平台 ({hostname})", css=CSS, elem_classes="product-shell")

    with blocks as demo:
        with gr.Column(elem_classes="product-header"):
            title = gr.HTML(padding=False)
            subtitle = gr.HTML(padding=False)

        engine.manager.add_elems("head", {"title": title, "subtitle": subtitle})
        with gr.Column(elem_classes=["surface-card", "model-config-card"]):
            engine.manager.add_elems("top", create_top())
        lang: gr.Dropdown = engine.manager.get_elem_by_id("top.lang")

        with gr.Tabs(elem_classes="product-tabs"):
            with gr.Tab("训练"):
                with gr.Column(elem_classes="surface-card"):
                    engine.manager.add_elems("train", create_train_tab(engine))

            with gr.Tab("评估与预测"):
                with gr.Column(elem_classes="surface-card"):
                    engine.manager.add_elems("eval", create_eval_tab(engine))

            with gr.Tab("对话验证"):
                with gr.Column(elem_classes="surface-card"):
                    engine.manager.add_elems("infer", create_infer_tab(engine))

            with gr.Tab("合并导出"):
                with gr.Column(elem_classes="surface-card"):
                    engine.manager.add_elems("export", create_export_tab(engine))

        engine.manager.add_elems("footer", create_footer())
        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None).then(
            engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False
        )
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
