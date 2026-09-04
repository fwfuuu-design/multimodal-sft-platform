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

from typing import TYPE_CHECKING

from ...data import TEMPLATES
from ...extras.constants import SUPPORTED_MODELS
from ...extras.misc import use_modelscope
from ...extras.packages import is_gradio_available
from ..common import save_config
from ..control import check_template, get_model_info, list_checkpoints, switch_hub
from ..mvp_client import MvpWebApiClient, MvpWebApiError, WebRequestContext, compute_node_view


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def refresh_runtime_context(request: "gr.Request") -> tuple["gr.Dropdown", str, str]:
    try:
        context = WebRequestContext.from_headers(request.headers)
        with MvpWebApiClient() as client:
            identity = client.get_identity(context)
            nodes = client.list_compute_nodes(context)
        choices, summary = compute_node_view(nodes)
        display_name = identity.get("display_name") or identity.get("user_id") or "已登录用户"
        role = identity.get("role") or "unknown"
        project_id = identity.get("project_id") or context.project_id or "未选择"
        identity_text = f"**{display_name}** · {role} · 项目 `{project_id}`"
        return gr.Dropdown(choices=choices, value=None, interactive=False), identity_text, summary
    except MvpWebApiError as error:
        return (
            gr.Dropdown(choices=[], value=None, interactive=False),
            "**未连接训练服务**",
            f"{error.message}（{error.code}）",
        )


def create_top() -> dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(choices=["zh", "en"], value="zh", scale=1)
        compatible_models = [
            name
            for name in SUPPORTED_MODELS
            if not any(token in name.lower() for token in ("audio", "omni", "video"))
        ]
        available_models = compatible_models + ["Custom"]
        model_name = gr.Dropdown(choices=available_models, value=None, scale=3)
        model_path = gr.Textbox(scale=3)
        default_hub = "modelscope" if use_modelscope() else "huggingface"
        hub_name = gr.Dropdown(choices=["local", "huggingface", "modelscope"], value=default_hub, scale=2)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=["lora"], value="lora", interactive=False, scale=1)
        checkpoint_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        quantization_bit = gr.Dropdown(choices=["none", "4"], value="none", allow_custom_value=False, scale=1)
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value="default", scale=2)

    with gr.Row(elem_classes="runtime-context"):
        runtime_identity = gr.Markdown("**未连接训练服务**")
        compute_node = gr.Dropdown(choices=[], value=None, interactive=False, scale=3)
        node_refresh = gr.Button(value="刷新节点", scale=1)
    node_status = gr.Markdown("无 GPU 服务器也可以接入，但不能领取训练任务。")

    node_refresh.click(refresh_runtime_context, outputs=[compute_node, runtime_identity, node_status], queue=False)

    model_name.change(get_model_info, [model_name], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    ).then(check_template, [lang, template])
    model_name.input(save_config, inputs=[lang, hub_name, model_name], queue=False)
    model_path.input(save_config, inputs=[lang, hub_name, model_name, model_path], queue=False)
    finetuning_type.change(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)
    hub_name.change(switch_hub, inputs=[hub_name], queue=False).then(
        get_model_info, [model_name], [model_path, template], queue=False
    ).then(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False).then(
        check_template, [lang, template]
    )
    hub_name.input(save_config, inputs=[lang, hub_name], queue=False)

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        hub_name=hub_name,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
        quantization_bit=quantization_bit,
        template=template,
        runtime_identity=runtime_identity,
        compute_node=compute_node,
        node_refresh=node_refresh,
        node_status=node_status,
    )
