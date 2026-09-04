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

"""Small registry used only by retained hardware kernel plugins."""

from abc import ABC
from typing import Any

from ...extras import logging


logger = logging.get_logger(__name__)


def ensure_methods_implemented(cls: type) -> None:
    required: set[str] = set()
    for base in cls.__mro__[1:]:
        required |= getattr(base, "__abstractmethods__", frozenset())

    missing = sorted(name for name in required if getattr(getattr(cls, name, None), "__isabstractmethod__", False))
    if missing:
        raise TypeError(f"{cls.__name__} does not implement all required methods: {missing}")


class BasePlugin(ABC):
    _registry: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def register(self):
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        cls = type(self)

        def decorator(obj: Any) -> Any:
            cls._registry[self.name] = obj
            return obj

        return decorator

    def _resolve(self) -> Any:
        cls = type(self)
        if self.name is None or self.name not in cls._registry:
            raise ValueError(f"Kernel plugin {self.name!r} is not registered.")
        return cls._registry[self.name]

    def __call__(self, *args, **kwargs) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._resolve(), attr)
