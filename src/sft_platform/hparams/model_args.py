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

"""Model, image processor, QLoRA, and Adapter-export arguments."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from omegaconf import OmegaConf

from ..extras.constants import AttentionFunction, EngineName, QuantizationMethod
from ..extras.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Local model path or a Hugging Face/ModelScope model identifier."},
    )
    adapter_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path or Hub identifier of one LoRA/QLoRA Adapter."},
    )
    adapter_folder: str | None = field(default=None, metadata={"help": "Adapter subfolder."})
    cache_dir: str | None = field(default=None, metadata={"help": "Model download cache directory."})
    model_revision: str = field(default="main", metadata={"help": "Model revision."})
    hf_hub_token: str | None = field(default=None, metadata={"help": "Hugging Face credential."})
    ms_hub_token: str | None = field(default=None, metadata={"help": "ModelScope credential."})
    om_hub_token: str | None = field(default=None, metadata={"help": "Modelers credential."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Allow model repository Python code."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use a fast tokenizer when available."})
    split_special_tokens: bool = field(default=False, metadata={"help": "Split special tokens during tokenization."})
    resize_vocab: bool = field(default=False, metadata={"help": "Resize tokenizer and model embeddings."})
    add_tokens: str | None = field(default=None, metadata={"help": "Comma-separated regular tokens to add."})
    add_special_tokens: str | None = field(default=None, metadata={"help": "Comma-separated special tokens to add."})
    new_special_tokens_config: str | None = field(
        default=None,
        metadata={"help": "YAML mapping from new special tokens to semantic descriptions."},
    )
    init_special_tokens: Literal["noise_init", "desc_init", "desc_init_w_noise"] = field(
        default="noise_init",
        metadata={"help": "Initialization method for newly added special tokens."},
    )
    low_cpu_mem_usage: bool = field(default=True, metadata={"help": "Use memory-efficient model loading."})
    flash_attn: AttentionFunction = field(
        default=AttentionFunction.AUTO,
        metadata={"help": "Attention implementation used by compatible model architectures."},
    )
    moe_aux_loss_coef: float | None = field(default=None, metadata={"help": "Optional native MoE router loss."})
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Disable gradient checkpointing."},
    )
    use_reentrant_gc: bool = field(default=True, metadata={"help": "Use reentrant gradient checkpointing."})
    upcast_layernorm: bool = field(default=False, metadata={"help": "Keep layer normalization in float32."})
    upcast_lmhead_output: bool = field(default=False, metadata={"help": "Return LM-head output in float32."})
    infer_backend: Literal[EngineName.HF] = field(
        default=EngineName.HF,
        metadata={"help": "Inference backend; only Hugging Face is supported."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Inference/export dtype."},
    )
    offload_folder: str = field(default="offload", metadata={"help": "Controlled model offload directory."})
    use_kv_cache: bool = field(default=True, metadata={"help": "Use KV cache during generation."})
    use_hardware_kernels: bool = field(
        default=False,
        metadata={"help": "Enable retained CUDA/NPU model kernels."},
    )
    print_param_status: bool = field(default=False, metadata={"help": "Print model parameter status for debugging."})

    quantization_method: Literal[QuantizationMethod.BNB] = field(
        default=QuantizationMethod.BNB,
        metadata={"help": "Quantization method; only bitsandbytes is supported."},
    )
    quantization_bit: Literal[4] | None = field(
        default=None,
        metadata={"help": "Set to 4 for QLoRA, or omit for LoRA."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(default="nf4", metadata={"help": "4-bit data type."})
    double_quantization: bool = field(default=True, metadata={"help": "Use nested 4-bit quantization."})
    quantization_device_map: Literal["auto"] | None = field(
        default=None,
        metadata={"help": "Automatic QLoRA device map for inference only."},
    )

    image_max_pixels: int = field(default=768 * 768, metadata={"help": "Maximum image pixels."})
    image_min_pixels: int = field(default=32 * 32, metadata={"help": "Minimum image pixels."})
    image_do_pan_and_scan: bool = field(default=False, metadata={"help": "Enable image pan-and-scan where supported."})
    crop_to_patches: bool = field(default=False, metadata={"help": "Crop images into patches where required."})

    export_dir: str | None = field(default=None, metadata={"help": "Directory for the merged model."})
    export_size: int = field(default=5, metadata={"help": "Maximum exported shard size in GB."})
    export_device: Literal["cpu", "auto"] = field(default="cpu", metadata={"help": "Adapter merge device."})

    compute_dtype: torch.dtype | None = field(default=None, init=False)
    device_map: str | dict[str, Any] | None = field(default=None, init=False)
    model_max_length: int | None = field(default=None, init=False)
    block_diag_attn: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.model_name_or_path is None:
            raise ValueError("Provide `model_name_or_path`.")
        if self.adapter_name_or_path is not None:
            adapters = [path.strip() for path in self.adapter_name_or_path.split(",") if path.strip()]
            if len(adapters) != 1:
                raise ValueError("Exactly one LoRA/QLoRA Adapter may be configured.")
            self.adapter_name_or_path = adapters
        if self.add_tokens is not None:
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",") if token.strip()]

        self._special_token_descriptions = None
        if self.new_special_tokens_config is not None:
            descriptions = OmegaConf.to_container(OmegaConf.load(self.new_special_tokens_config))
            if not isinstance(descriptions, dict):
                raise ValueError("The special-token YAML must map tokens to descriptions.")
            self._special_token_descriptions = descriptions
            self.add_special_tokens = list(descriptions)
        elif self.add_special_tokens is not None:
            self.add_special_tokens = [token.strip() for token in self.add_special_tokens.split(",") if token.strip()]

        if self.init_special_tokens != "noise_init" and self._special_token_descriptions is None:
            logger.warning_rank0("Semantic token initialization needs descriptions; falling back to noise initialization.")
            self.init_special_tokens = "noise_init"
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError("`image_max_pixels` cannot be smaller than `image_min_pixels`.")
        if self.export_size <= 0:
            raise ValueError("`export_size` must be greater than zero.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {key: f"<{key.upper()}>" if key.endswith("token") else value for key, value in values.items()}
