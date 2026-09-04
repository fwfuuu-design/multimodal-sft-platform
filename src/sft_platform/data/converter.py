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

"""Convert Alpaca, ShareGPT, and OpenAI text/image SFT records."""

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from ..extras import logging
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import ImageInput
    from .parser import DatasetAttr


logger = logging.get_logger(__name__)


def _empty_media() -> dict[str, None]:
    """Keep the stable processor schema while product inputs remain image-only."""
    return {"_videos": None, "_audios": None}


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_images(self, images: Union["ImageInput", list["ImageInput"], None]) -> list["ImageInput"] | None:
        if images is None:
            return None
        if not isinstance(images, list):
            images = [images]
        if not images:
            return None
        images = images[:]
        if self.dataset_attr.load_from in {"script", "file"}:
            for index, image in enumerate(images):
                if not isinstance(image, str):
                    continue
                image_path = os.path.join(self.data_args.media_dir, image)
                if os.path.isfile(image_path):
                    images[index] = image_path
                else:
                    logger.warning_rank0_once(f"Image {image} does not exist in `media_dir`; using the original path.")
        return images

    def _output(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str,
        tools: str,
        example: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": tools,
            "_images": self._find_images(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            **_empty_media(),
        }

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt: list[dict[str, str]] = []
        if self.dataset_attr.history and isinstance(example.get(self.dataset_attr.history), list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.extend(
                    [
                        {"role": Role.USER.value, "content": old_prompt},
                        {"role": Role.ASSISTANT.value, "content": old_response},
                    ]
                )
        query = []
        if self.dataset_attr.prompt and example.get(self.dataset_attr.prompt):
            query.append(example[self.dataset_attr.prompt])
        if self.dataset_attr.query and example.get(self.dataset_attr.query):
            query.append(example[self.dataset_attr.query])
        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})

        response_text = example.get(self.dataset_attr.response) if self.dataset_attr.response else None
        response = (
            [{"role": Role.ASSISTANT.value, "content": response_text}] if isinstance(response_text, str) else []
        )
        system = example.get(self.dataset_attr.system, "") if self.dataset_attr.system else ""
        tools = example.get(self.dataset_attr.tools, "") if self.dataset_attr.tools else ""
        return self._output(prompt, response, system, tools, example)


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        messages = list(example[self.dataset_attr.messages])
        if messages and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag:
            system = messages.pop(0)[self.dataset_attr.content_tag]
        else:
            system = example.get(self.dataset_attr.system, "") if self.dataset_attr.system else ""

        expected = (
            (self.dataset_attr.user_tag, self.dataset_attr.observation_tag),
            (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag),
        )
        aligned = []
        for index, message in enumerate(messages):
            role = message[self.dataset_attr.role_tag]
            if role not in expected[index % 2]:
                logger.warning_rank0("Skipping an SFT example with invalid role order.")
                aligned = []
                break
            aligned.append({"role": tag_mapping[role], "content": message[self.dataset_attr.content_tag]})
        if len(aligned) % 2 != 0:
            logger.warning_rank0("Skipping an SFT example with an incomplete assistant response.")
            aligned = []

        tools = example.get(self.dataset_attr.tools, "") if self.dataset_attr.tools else ""
        return self._output(aligned[:-1], aligned[-1:], system, tools, example)


@dataclass
class OpenAIDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = list(example[self.dataset_attr.messages])
        if messages and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag:
            system = messages.pop(0)[self.dataset_attr.content_tag]
        else:
            system = example.get(self.dataset_attr.system, "") if self.dataset_attr.system else ""

        role_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
        }
        aligned = []
        observations = []
        for message in messages:
            role = message[self.dataset_attr.role_tag]
            content = message[self.dataset_attr.content_tag]
            if message.get("tool_calls"):
                content = json.dumps([item["function"] for item in message["tool_calls"]], ensure_ascii=False)
                role = self.dataset_attr.function_tag
            if role == self.dataset_attr.observation_tag:
                observations.append(content)
                continue
            if observations:
                aligned.append({"role": Role.OBSERVATION.value, "content": "\n".join(observations)})
                observations = []
            if role not in role_mapping:
                logger.warning_rank0("Skipping an SFT example with an unsupported message role.")
                aligned = []
                break
            aligned.append({"role": role_mapping[role], "content": content})
        if observations:
            aligned.append({"role": Role.OBSERVATION.value, "content": "\n".join(observations)})

        expected_roles = ((Role.USER.value, Role.OBSERVATION.value), (Role.ASSISTANT.value, Role.FUNCTION.value))
        if len(aligned) % 2 or any(message["role"] not in expected_roles[index % 2] for index, message in enumerate(aligned)):
            logger.warning_rank0("Skipping an SFT example with invalid message ordering.")
            aligned = []

        tools = example.get(self.dataset_attr.tools, "") if self.dataset_attr.tools else ""
        if isinstance(tools, (dict, list)):
            tools = json.dumps(tools, ensure_ascii=False)
        return self._output(aligned[:-1], aligned[-1:], system, tools, example)


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "openai": OpenAIDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")
    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")
    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = {
            "num_proc": data_args.preprocessing_num_workers,
            "load_from_cache_file": (not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            "desc": "Converting SFT dataset format",
        }
    converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    return dataset.map(converter, batched=False, remove_columns=column_names, **kwargs)
