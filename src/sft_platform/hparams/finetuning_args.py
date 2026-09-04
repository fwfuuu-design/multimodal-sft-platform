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

"""SFT LoRA/QLoRA parameters exposed by the product runtime."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FinetuningArguments:
    stage: Literal["sft"] = field(
        default="sft",
        metadata={"help": "Training stage. The product supports SFT only."},
    )
    finetuning_type: Literal["lora"] = field(
        default="lora",
        metadata={"help": "Fine-tuning method. The product supports LoRA/QLoRA only."},
    )
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int | None = field(default=None, metadata={"help": "LoRA scale; defaults to twice the rank."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout rate."})
    lora_target: str | list[str] = field(
        default="all",
        metadata={"help": "Comma-separated target modules, or `all` for every supported linear layer."},
    )
    additional_target: str | list[str] | set[str] | None = field(
        default=None,
        metadata={"help": "Additional trainable modules saved with the Adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Create a new Adapter instead of resuming the configured Adapter."},
    )
    pure_bf16: bool = field(default=False, metadata={"help": "Train in pure bfloat16 without AMP."})
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Freeze the vision tower during image-text SFT."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Freeze the multimodal projector during image-text SFT."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Freeze the language model while training selected multimodal targets."},
    )
    compute_accuracy: bool = field(default=False, metadata={"help": "Compute token-level evaluation accuracy."})
    disable_shuffling: bool = field(default=False, metadata={"help": "Disable training-set shuffling."})
    plot_loss: bool = field(default=False, metadata={"help": "Save the training loss curve."})
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Include effective tokens per second in training metrics."},
    )

    def __post_init__(self) -> None:
        if self.stage != "sft":
            raise ValueError("Only SFT training is available in this product.")
        if self.finetuning_type != "lora":
            raise ValueError("Only LoRA/QLoRA fine-tuning is available in this product.")
        if self.lora_rank <= 0:
            raise ValueError("`lora_rank` must be greater than zero.")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("`lora_dropout` must be in [0, 1).")

        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        if isinstance(self.lora_target, str):
            self.lora_target = [item.strip() for item in self.lora_target.split(",") if item.strip()]
        if isinstance(self.additional_target, str):
            self.additional_target = [item.strip() for item in self.additional_target.split(",") if item.strip()]
        if not self.lora_target:
            raise ValueError("At least one `lora_target` is required.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
