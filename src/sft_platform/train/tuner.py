# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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

"""SFT-only training and LoRA/QLoRA Adapter merge entrypoints."""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.mvp_policy import validate_mvp_runtime_environment, validate_mvp_train_config
from ..extras.packages import is_transformers_version_greater_than
from ..hparams import get_infer_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    if finetuning_args.stage != "sft":
        raise ValueError("Only supervised fine-tuning (SFT) is available in this product.")

    callbacks.append(LogCallback())
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as error:
        logger.warning(f"Failed to destroy process group: {error}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    validate_mvp_runtime_environment()
    validate_mvp_train_config(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    _training_function(config={"args": args, "callbacks": callbacks or []})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    """Merge one LoRA/QLoRA Adapter into its base model and write a standalone model directory."""
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")
    if model_args.adapter_name_or_path is None or len(model_args.adapter_name_or_path) != 1:
        raise ValueError("Exactly one LoRA/QLoRA Adapter is required for export.")
    if finetuning_args.finetuning_type != "lora":
        raise ValueError("Only LoRA/QLoRA Adapter merge is available for export.")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)

    if getattr(model, "quantization_method", None) is not None:
        raise ValueError("Load an unquantized base model before merging an Adapter.")
    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if model_args.infer_dtype == "auto":
        output_dtype = getattr(model.config, "torch_dtype", torch.float32)
        if output_dtype == torch.float32:
            output_dtype = infer_optim_dtype(torch.bfloat16)
    else:
        output_dtype = getattr(torch, model_args.infer_dtype)

    setattr(model.config, "torch_dtype", output_dtype)
    model = model.to(output_dtype)
    logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    save_kwargs = {"save_directory": model_args.export_dir, "max_shard_size": f"{model_args.export_size}GB"}
    if not is_transformers_version_greater_than("5.0.0"):
        save_kwargs["safe_serialization"] = True

    try:
        model.save_pretrained(**save_kwargs)
    except NotImplementedError as error:
        raise RuntimeError(
            "Failed to export model because weight conversion reversal is unsupported for this architecture."
        ) from error

    try:
        tokenizer.padding_side = "left"
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
    except Exception as error:
        logger.warning_rank0(f"Cannot save tokenizer or processor: {error}.")
