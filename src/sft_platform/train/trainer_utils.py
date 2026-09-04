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

"""Small trainer helpers required by the SFT workflow."""

from typing import TYPE_CHECKING

from transformers import Trainer


if TYPE_CHECKING:
    from ..hparams import DataArguments, FinetuningArguments, ModelArguments, TrainingArguments


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    """Write a local model card without publishing weights to an external Hub."""
    if not training_args.do_train:
        return

    metadata = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["multimodal-sft-platform", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        metadata["dataset"] = data_args.dataset
    Trainer.create_model_card(trainer, license="other", **metadata)
