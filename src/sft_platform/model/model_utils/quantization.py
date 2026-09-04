# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

"""Unquantized LoRA and 4-bit bitsandbytes QLoRA loading."""

from typing import TYPE_CHECKING, Any

from transformers import BitsAndBytesConfig

from ...extras import logging
from ...extras.constants import QuantizationMethod
from ...extras.misc import check_version, get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    init_kwargs: dict[str, Any],
) -> None:
    del tokenizer, is_trainable
    if getattr(config, "quantization_config", None):
        raise ValueError("Pre-quantized base models are outside the product scope; use a standard base model.")
    if model_args.quantization_bit is None:
        return
    if model_args.quantization_bit != 4 or model_args.quantization_method != QuantizationMethod.BNB:
        raise ValueError("Only 4-bit bitsandbytes QLoRA is available in this product.")

    check_version("bitsandbytes>=0.43.0", mandatory=True)
    init_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=model_args.compute_dtype,
        bnb_4bit_use_double_quant=model_args.double_quantization,
        bnb_4bit_quant_type=model_args.quantization_type,
        bnb_4bit_quant_storage=model_args.compute_dtype,
    )
    if model_args.quantization_device_map != "auto":
        init_kwargs["device_map"] = {"": get_current_device()}
    logger.info_rank0("Quantizing the base model to 4-bit with bitsandbytes for QLoRA.")
