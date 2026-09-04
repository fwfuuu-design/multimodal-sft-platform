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

"""SFT-only Train tab for text and image-text LoRA/QLoRA jobs."""

from typing import TYPE_CHECKING

from transformers.trainer_utils import SchedulerType

from ...extras.misc import get_device_count
from ...extras.packages import is_gradio_available
from ..common import DEFAULT_DATA_DIR
from ..control import change_stage, list_config_paths, list_datasets, list_output_dirs
from .data import create_preview_box


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


SFT_STAGE_NAME = "Supervised Fine-Tuning"


def create_train_tab(engine: "Engine") -> dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict: dict[str, Component] = {}

    with gr.Row():
        training_stage = gr.Dropdown(
            choices=[SFT_STAGE_NAME], value=SFT_STAGE_NAME, interactive=False, scale=1
        )
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=1)
        dataset = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    input_elems.update({training_stage, dataset_dir, dataset})
    elem_dict.update(dict(training_stage=training_stage, dataset_dir=dataset_dir, dataset=dataset, **preview_elems))

    with gr.Row():
        learning_rate = gr.Textbox(value="5e-5")
        num_train_epochs = gr.Textbox(value="3.0")
        max_grad_norm = gr.Textbox(value="1.0")
        train_seed = gr.Textbox(value="42")
        max_samples = gr.Textbox(value="100000")
        compute_type = gr.Dropdown(choices=["bf16", "fp16", "fp32", "pure_bf16"], value="bf16")

    input_elems.update({learning_rate, num_train_epochs, max_grad_norm, train_seed, max_samples, compute_type})
    elem_dict.update(
        dict(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_grad_norm=max_grad_norm,
            train_seed=train_seed,
            max_samples=max_samples,
            compute_type=compute_type,
        )
    )

    with gr.Row():
        cutoff_len = gr.Slider(minimum=4, maximum=131072, value=2048, step=1)
        batch_size = gr.Slider(minimum=1, maximum=1024, value=2, step=1)
        gradient_accumulation_steps = gr.Slider(minimum=1, maximum=1024, value=8, step=1)
        val_size = gr.Slider(minimum=0, maximum=1, value=0, step=0.001)
        lr_scheduler_type = gr.Dropdown(
            choices=[scheduler.value for scheduler in SchedulerType], value="cosine"
        )

    input_elems.update({cutoff_len, batch_size, gradient_accumulation_steps, val_size, lr_scheduler_type})
    elem_dict.update(
        dict(
            cutoff_len=cutoff_len,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            val_size=val_size,
            lr_scheduler_type=lr_scheduler_type,
        )
    )

    with gr.Accordion(open=False) as extra_tab:
        with gr.Row():
            logging_steps = gr.Slider(minimum=1, maximum=1000, value=5, step=5)
            save_steps = gr.Slider(minimum=10, maximum=5000, value=100, step=10)
            warmup_steps = gr.Slider(minimum=0, maximum=5000, value=0, step=1)
        with gr.Row():
            packing = gr.Checkbox()
            neat_packing = gr.Checkbox()
            train_on_prompt = gr.Checkbox()
            mask_history = gr.Checkbox()
            resize_vocab = gr.Checkbox()
            enable_thinking = gr.Checkbox(value=True)

    input_elems.update(
        {logging_steps, save_steps, warmup_steps, packing, neat_packing, train_on_prompt, mask_history, resize_vocab, enable_thinking}
    )
    elem_dict.update(
        dict(
            extra_tab=extra_tab,
            logging_steps=logging_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            packing=packing,
            neat_packing=neat_packing,
            train_on_prompt=train_on_prompt,
            mask_history=mask_history,
            resize_vocab=resize_vocab,
            enable_thinking=enable_thinking,
        )
    )

    with gr.Accordion(open=False) as lora_tab:
        with gr.Row():
            lora_rank = gr.Slider(minimum=1, maximum=1024, value=8, step=1)
            lora_alpha = gr.Slider(minimum=1, maximum=2048, value=16, step=1)
            lora_dropout = gr.Slider(minimum=0, maximum=1, value=0, step=0.01)
            create_new_adapter = gr.Checkbox()
        with gr.Row():
            lora_target = gr.Textbox(scale=2)
            additional_target = gr.Textbox(scale=2)

    input_elems.update({lora_rank, lora_alpha, lora_dropout, create_new_adapter, lora_target, additional_target})
    elem_dict.update(
        dict(
            lora_tab=lora_tab,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            create_new_adapter=create_new_adapter,
            lora_target=lora_target,
            additional_target=additional_target,
        )
    )

    with gr.Accordion(open=False) as mm_tab:
        with gr.Row():
            freeze_vision_tower = gr.Checkbox(value=True)
            freeze_multi_modal_projector = gr.Checkbox(value=True)
            freeze_language_model = gr.Checkbox(value=False)
        with gr.Row():
            image_max_pixels = gr.Textbox(value="768*768")
            image_min_pixels = gr.Textbox(value="32*32")

    input_elems.update(
        {freeze_vision_tower, freeze_multi_modal_projector, freeze_language_model, image_max_pixels, image_min_pixels}
    )
    elem_dict.update(
        dict(
            mm_tab=mm_tab,
            freeze_vision_tower=freeze_vision_tower,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            freeze_language_model=freeze_language_model,
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
        )
    )

    with gr.Row():
        cmd_preview_btn = gr.Button()
        arg_save_btn = gr.Button()
        arg_load_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                current_time = gr.Textbox(visible=False, interactive=False)
                output_dir = gr.Dropdown(allow_custom_value=True)
                config_path = gr.Dropdown(allow_custom_value=True)
            with gr.Row():
                device_count = gr.Textbox(value=str(get_device_count() or 1), interactive=False)
            with gr.Row():
                resume_btn = gr.Checkbox(visible=False, interactive=False)
                progress_bar = gr.Slider(visible=False, interactive=False)
            with gr.Row():
                output_box = gr.Markdown()
        with gr.Column(scale=1):
            loss_viewer = gr.Plot()

    input_elems.update({output_dir, config_path})
    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            arg_save_btn=arg_save_btn,
            arg_load_btn=arg_load_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            current_time=current_time,
            output_dir=output_dir,
            config_path=config_path,
            device_count=device_count,
            resume_btn=resume_btn,
            progress_bar=progress_bar,
            output_box=output_box,
            loss_viewer=loss_viewer,
        )
    )
    output_elems = [output_box, progress_bar, loss_viewer]

    cmd_preview_btn.click(engine.runner.preview_train, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_train, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)
    resume_btn.change(engine.runner.monitor, outputs=output_elems, concurrency_limit=None)

    lang = engine.manager.get_elem_by_id("top.lang")
    model_name: gr.Dropdown = engine.manager.get_elem_by_id("top.model_name")
    finetuning_type: gr.Dropdown = engine.manager.get_elem_by_id("top.finetuning_type")

    arg_save_btn.click(engine.runner.save_args, input_elems, output_elems, concurrency_limit=None)
    arg_load_btn.click(
        engine.runner.load_args, [lang, config_path], list(input_elems) + [output_box], concurrency_limit=None
    )
    dataset.focus(list_datasets, [dataset_dir, training_stage], [dataset], queue=False)
    training_stage.change(change_stage, [training_stage], [dataset, packing], queue=False)
    model_name.change(list_output_dirs, [model_name, finetuning_type, current_time], [output_dir], queue=False)
    finetuning_type.change(list_output_dirs, [model_name, finetuning_type, current_time], [output_dir], queue=False)
    output_dir.change(
        list_output_dirs, [model_name, finetuning_type, current_time], [output_dir], concurrency_limit=None
    )
    output_dir.input(
        engine.runner.check_output_dir,
        [lang, model_name, finetuning_type, output_dir],
        list(input_elems) + [output_box],
        concurrency_limit=None,
    )
    config_path.change(list_config_paths, [current_time], [config_path], queue=False)

    return elem_dict
