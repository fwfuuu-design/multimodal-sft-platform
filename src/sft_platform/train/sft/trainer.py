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

"""SFT-only Seq2SeqTrainer extension."""

import json
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Seq2SeqTrainer with multimodal processor saving and generation output."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))
        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()
        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        if not self.is_world_process_zero():
            return

        output_path = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_path}")
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        predictions = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )
        for index in range(len(predictions)):
            non_padding = np.nonzero(predictions[index] != self.processing_class.pad_token_id)[0]
            if len(non_padding):
                predictions[index] = np.concatenate(
                    (predictions[index][non_padding[0] :], predictions[index][: non_padding[0]]), axis=-1
                )

        input_ids = dataset["input_ids"]
        try:
            input_ids = input_ids.to_pylist()
        except AttributeError:
            input_ids = list(input_ids)
        decoded_inputs = self.processing_class.batch_decode(input_ids, skip_special_tokens=False)
        decoded_predictions = self.processing_class.batch_decode(predictions, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_path, "w", encoding="utf-8") as file:
            for prompt, prediction, label in zip(decoded_inputs, decoded_predictions, decoded_labels):
                file.write(
                    json.dumps({"prompt": prompt, "predict": prediction, "label": label}, ensure_ascii=False) + "\n"
                )
