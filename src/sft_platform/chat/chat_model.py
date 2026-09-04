# Copyright 2025 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
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

"""Hugging Face-only text and image-text Chat facade."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from ..extras.constants import EngineName
from ..extras.misc import torch_gc
from ..hparams import get_infer_args


if TYPE_CHECKING:
    from ..data.mm_plugin import ImageInput
    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    r"""Synchronous and asynchronous Chat API backed only by Hugging Face."""

    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        if model_args.infer_backend != EngineName.HF:
            raise ValueError("Only the Hugging Face inference backend is available in this product.")

        from .hf_engine import HuggingfaceEngine

        self.engine: BaseEngine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, tools, images, **input_kwargs), self._loop
        )
        return task.result()

    async def achat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        return await self.engine.chat(messages, system, tools, images, **input_kwargs)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, system, tools, images, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, system, tools, images, **input_kwargs):
            yield new_token

    def __del__(self) -> None:
        torch_gc()
