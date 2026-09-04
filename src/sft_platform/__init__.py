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

r"""Text and image-text SFT with LoRA/QLoRA.

Dependency layers:
  api, webui, cli > mvp, train, chat > data, model > hparams, extras

Set logging verbosity with SFT_PLATFORM_VERBOSITY=WARN.
"""

from .extras.env import VERSION


__version__ = VERSION
