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

import torch
from torch import nn
from torch.nn import functional as F

from ...extras.packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from torch import nn
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


def configure_moe(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.moe_aux_loss_coef:
        return

    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", None)  # for multimodal model

    if model_type in [
        "dbrx",
        "ernie4_5_moe",
        "granitemoe",
        "jamba",
        "jetmoe",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "output_router_logits", True)

    if text_config and getattr(text_config, "model_type", None) in [
        "glm4v_moe_text",  # glmv4_5
        "qwen3_moe",  # internvl_3_5
    ]:
        setattr(text_config, "output_router_logits", True)

    if model_type in [
        "ernie4_5_moe",
        "granitemoe",
        "jamba",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    elif text_config and getattr(text_config, "model_type", None) in ["qwen3_moe"]:
        setattr(text_config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    elif model_type == "deepseek":
        setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

    elif model_type == "jetmoe":
        setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)


class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # Calculate the routing weights for all experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Retain the weight of the top_k and reset the rest of the expert rights to 0 (instead of retaining only top_k experts)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # Initialize the all-zero weight matrix (same shape as all experts)
        full_routing_weights = torch.zeros_like(routing_weights)
        # Only the weight of top_k experts is retained, and the weight of the rest of the experts remains at 0
        full_routing_weights.scatter_(1, top_k_indices, top_k_weights)

        # Normalized top_k weights (keep the original logic consistent)
        if self.norm_topk_prob:
            # Calculate the sum of the weights top_k each row (for normalization)
            top_k_sum = full_routing_weights.sum(dim=-1, keepdim=True)
            # Avoid dividing by zero
            top_k_sum = torch.clamp(top_k_sum, min=1e-9)
            full_routing_weights /= top_k_sum

        # Convert back to the input data type
        full_routing_weights = full_routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # Go through all the experts (not just the selected ones)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Get the weight of the current expert (inactive expert has a weight of 0 here)
            expert_weights = full_routing_weights[:, expert_idx, None]  # shape: (batch*seq, 1)
            # All samples participate in the calculations of the current expert, the weight may be equal to 0
            current_hidden_states = expert_layer(hidden_states) * expert_weights
            # Add-up to all expert outputs (experts with a weight of 0 do not affect the result)
            final_hidden_states += current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
