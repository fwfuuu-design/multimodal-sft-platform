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

"""Optional dependency checks used by retained product and hardware paths."""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

import transformers.utils.import_utils as import_utils
from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_pyav_available():
    return _is_package_available("av")


def is_jieba_available():
    return _is_package_available("jieba")


def is_gradio_available():
    return _is_package_available("gradio")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_pillow_available():
    return _is_package_available("PIL")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


@lru_cache
def is_transformers_version_greater_than(content: str):
    return _get_package_version("transformers") >= version.parse(content)


@lru_cache
def is_torch_version_greater_than(content: str):
    return _get_package_version("torch") >= version.parse(content)


_original_is_package_available = import_utils._is_package_available


class PackageAvailability(tuple):
    __slots__ = ()

    def __new__(cls, available: bool, package_version: str = "N/A"):
        return super().__new__(cls, (bool(available), package_version))

    def __bool__(self) -> bool:
        return self[0]


def _patched_is_package_available(package_name: str, return_version: bool = False):
    available, package_version = _original_is_package_available(package_name, return_version=return_version)
    return PackageAvailability(available, package_version)


if is_transformers_version_greater_than("5.3.0"):
    import_utils._is_package_available = _patched_is_package_available
