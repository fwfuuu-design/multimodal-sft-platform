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

"""LoRA/QLoRA-only Adapter initialization and loading."""

from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from ..extras import logging
from .model_utils.misc import find_all_linear_modules
from .model_utils.visual import patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _adapter_load_kwargs(model_args: "ModelArguments") -> dict[str, object]:
    return {
        "subfolder": model_args.adapter_folder,
        "offload_folder": model_args.offload_folder,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def _load_existing_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> tuple["PreTrainedModel", bool]:
    adapter_paths = model_args.adapter_name_or_path
    if not adapter_paths:
        return model, False
    if len(adapter_paths) != 1:
        raise ValueError("The product accepts exactly one LoRA/QLoRA Adapter at a time.")

    adapter_path = adapter_paths[0]
    quantized = getattr(model, "quantization_method", None) is not None
    should_resume = is_trainable and not finetuning_args.create_new_adapter
    loaded = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=should_resume,
        **_adapter_load_kwargs(model_args),
    )
    if not should_resume and not quantized:
        loaded = loaded.merge_and_unload()
        logger.info_rank0("Merged one LoRA Adapter into the base model.")
    else:
        logger.info_rank0(f"Loaded LoRA Adapter: {adapter_path}")
    return loaded, should_resume


def _create_lora_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> "PeftModel":
    if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
        target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
    else:
        target_modules = finetuning_args.lora_target
    target_modules = patch_target_modules(model, finetuning_args, target_modules)

    if model_args.resize_vocab and finetuning_args.additional_target is None:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        module_names = {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if module in [input_embeddings, output_embeddings]
        }
        finetuning_args.additional_target = module_names
        logger.warning_rank0("Vocab was resized; embedding modules were added to trainable targets.")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetuning_args.lora_rank,
        target_modules=target_modules,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        modules_to_save=finetuning_args.additional_target,
    )
    return get_peft_model(model, config)


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize, resume, or merge one LoRA/QLoRA Adapter."""
    del config  # kept in the stable loader signature
    if finetuning_args.finetuning_type != "lora":
        raise ValueError("Only LoRA/QLoRA fine-tuning is available in this product.")

    if is_trainable:
        logger.info_rank0("Fine-tuning method: LoRA" if model_args.quantization_bit is None else "QLoRA")

    model, resumed = _load_existing_adapter(model, model_args, finetuning_args, is_trainable)
    if is_trainable and not resumed:
        model = _create_lora_adapter(model, model_args, finetuning_args)

    if is_trainable and not finetuning_args.pure_bf16:
        logger.info_rank0("Upcasting trainable params to float32.")
        for parameter in filter(lambda item: item.requires_grad, model.parameters()):
            parameter.data = parameter.data.to(torch.float32)

    return model
