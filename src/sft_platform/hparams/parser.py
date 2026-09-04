# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace transformers library.
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

"""Argument parsing for the single SFT/LoRA product runtime."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES, EngineName, QuantizationMethod
from ..extras.misc import check_dependencies, check_version, get_current_device, is_env_enabled
from ..extras.mvp_policy import validate_mvp_runtime_environment, validate_mvp_train_config
from .data_args import DataArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments


logger = logging.get_logger(__name__)
check_dependencies()

_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]
_INFER_CLS = tuple[ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments]


def read_args(args: dict[str, Any] | list[str] | None = None) -> dict[str, Any] | list[str]:
    r"""Read explicit arguments, YAML/JSON config, or the current command line."""
    if args is not None:
        return args
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        overrides = OmegaConf.from_cli(sys.argv[2:])
        config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(config, overrides))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        overrides = OmegaConf.from_cli(sys.argv[2:])
        config = OmegaConf.create(json.load(Path(sys.argv[1]).absolute()))
        return OmegaConf.to_container(OmegaConf.merge(config, overrides))
    return sys.argv[1:]


def _parse_args(
    parser: HfArgumentParser,
    args: dict[str, Any] | list[str],
    *,
    allow_extra_keys: bool = False,
) -> tuple[Any, ...]:
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)
    (*parsed, unknown) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    if unknown and not allow_extra_keys:
        raise ValueError(f"Unsupported command arguments: {unknown}")
    return tuple(parsed)


def _set_transformers_logging() -> None:
    if os.getenv("SFT_PLATFORM_VERBOSITY", "INFO") in {"DEBUG", "INFO"}:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _set_hardware_environment() -> None:
    if is_torch_npu_available():
        torch.npu.set_compile_mode(jit_compile=is_env_enabled("NPU_JIT_COMPILE"))


def _verify_model_args(model_args: ModelArguments, finetuning_args: FinetuningArguments) -> None:
    if finetuning_args.stage != "sft" or finetuning_args.finetuning_type != "lora":
        raise ValueError("Only SFT with LoRA/QLoRA is available in this product.")
    if model_args.infer_backend != EngineName.HF:
        raise ValueError("Only the Hugging Face inference backend is available in this product.")
    if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
        raise ValueError("Exactly one LoRA/QLoRA Adapter may be loaded at a time.")
    if model_args.quantization_bit not in {None, 4}:
        raise ValueError("Only unquantized LoRA or 4-bit QLoRA is available in this product.")
    if model_args.quantization_bit == 4 and model_args.quantization_method != QuantizationMethod.BNB:
        raise ValueError("4-bit QLoRA requires the bitsandbytes quantization method.")
    if model_args.quantization_bit is not None and model_args.resize_vocab:
        raise ValueError("A quantized model cannot resize embedding layers.")


def _check_dependencies(
    finetuning_args: FinetuningArguments,
    training_args: TrainingArguments | None = None,
) -> None:
    if finetuning_args.plot_loss:
        check_version("matplotlib", mandatory=True)
    if training_args is not None and training_args.predict_with_generate:
        check_version("jieba", mandatory=True)
        check_version("nltk", mandatory=True)
        check_version("rouge_chinese", mandatory=True)


def _parse_train_args(args: dict[str, Any] | list[str]) -> _TRAIN_CLS:
    return _parse_args(HfArgumentParser(_TRAIN_ARGS), args, allow_extra_keys=is_env_enabled("ALLOW_EXTRA_ARGS"))


def _parse_infer_args(args: dict[str, Any] | list[str]) -> _INFER_CLS:
    return _parse_args(HfArgumentParser(_INFER_ARGS), args, allow_extra_keys=is_env_enabled("ALLOW_EXTRA_ARGS"))


def get_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    raw_args = read_args(args)
    validate_mvp_runtime_environment()
    validate_mvp_train_config(raw_args)
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(raw_args)

    if training_args.should_log:
        _set_transformers_logging()
    _set_hardware_environment()
    _verify_model_args(model_args, finetuning_args)
    _check_dependencies(finetuning_args, training_args)

    if training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Enable `predict_with_generate` to save prediction output.")
    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Specify `max_steps` when using a streaming dataset.")
    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Specify a dataset for SFT training.")
    if (training_args.do_eval or training_args.do_predict) and data_args.eval_dataset is None and data_args.val_size < 1e-6:
        raise ValueError("Provide `eval_dataset` or set `val_size` above zero.")
    if training_args.predict_with_generate and finetuning_args.compute_accuracy:
        raise ValueError("Generation metrics and token accuracy cannot be enabled together.")
    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Automatic device maps are not available for QLoRA training.")
    if finetuning_args.pure_bf16 and not (
        is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())
    ):
        raise ValueError("This device does not support pure bfloat16 training.")

    if training_args.do_train and model_args.quantization_bit is not None and not model_args.upcast_layernorm:
        logger.warning_rank0("Enabling `upcast_layernorm` is recommended for QLoRA training.")
    if training_args.do_train and not training_args.fp16 and not training_args.bf16:
        logger.warning_rank0("Mixed precision training is recommended.")

    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False
    training_args.label_names = training_args.label_names or ["labels"]
    if training_args.parallel_mode == ParallelMode.DISTRIBUTED and training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = False

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name)) for name in CHECKPOINT_NAMES
        ):
            raise ValueError("Output directory is not empty; enable `overwrite_output_dir` or choose another path.")
        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info_rank0(f"Resuming training from {last_checkpoint}.")

    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
    data_args.packing = bool(data_args.packing)
    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    transformers.set_seed(training_args.seed)
    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: dict[str, Any] | list[str] | None = None) -> _INFER_CLS:
    raw_args = read_args(args)
    validate_mvp_runtime_environment()
    validate_mvp_train_config(raw_args)
    model_args, data_args, finetuning_args, generating_args = _parse_infer_args(raw_args)
    _set_transformers_logging()
    _set_hardware_environment()
    _verify_model_args(model_args, finetuning_args)
    _check_dependencies(finetuning_args)

    if model_args.export_dir is not None and model_args.export_device == "cpu":
        model_args.device_map = {"": torch.device("cpu")}
        if data_args.cutoff_len != DataArguments().cutoff_len:
            model_args.model_max_length = data_args.cutoff_len
    else:
        model_args.device_map = "auto"
    return model_args, data_args, finetuning_args, generating_args
