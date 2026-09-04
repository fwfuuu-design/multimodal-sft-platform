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


def main():
    import sys

    from .extras.mvp_policy import validate_mvp_runtime_environment

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        validate_mvp_runtime_environment()

    from . import launcher

    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
