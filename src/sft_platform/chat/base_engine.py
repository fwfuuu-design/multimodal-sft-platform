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

"""Text and image-text Chat engine contracts."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ..data import Template
    from ..data.mm_plugin import ImageInput
    from ..extras.constants import EngineName
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class BaseEngine(ABC):
    name: "EngineName"
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizer"
    template: "Template"
    generating_args: dict[str, Any]

    @abstractmethod
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None: ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]: ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]: ...
