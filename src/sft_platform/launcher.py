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

"""Public CLI dispatcher for the SFT-only product surface."""

import os
import subprocess
import sys
from copy import deepcopy


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   sft-train api: launch the authenticated API                     |\n"
    + "|   sft-train bootstrap: create the initial project                 |\n"
    + "|   sft-train backup --dry-run: preview a backup                    |\n"
    + "|   sft-train restore --dry-run FILE: preview restore               |\n"
    + "|   sft-train train -h: train SFT models with LoRA/QLoRA           |\n"
    + "|   sft-train webui: launch the product Web UI                      |\n"
    + "|   sft-train env: show environment info                            |\n"
    + "|   sft-train version: show version info                            |\n"
    + "-" * 70
)


def _launch_single_machine_workers(device_count: int) -> None:
    from .extras import logging
    from .extras.misc import find_available_port, is_env_enabled

    logger = logging.get_logger(__name__)
    nproc_per_node = os.getenv("NPROC_PER_NODE", str(max(device_count, 1)))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", str(find_available_port()))
    logger.info_rank0(f"Initializing {nproc_per_node} local distributed tasks at: {master_addr}:{master_port}")

    env = deepcopy(os.environ)
    if is_env_enabled("OPTIM_TORCH", "1"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    process = subprocess.run(
        [
            "torchrun",
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--nproc_per_node",
            nproc_per_node,
            "--master_addr",
            master_addr,
            "--master_port",
            master_port,
            __file__,
            *sys.argv[1:],
        ],
        env=env,
        check=True,
    )
    raise SystemExit(process.returncode)


def launch() -> None:
    from .extras.env import VERSION, print_env
    from .extras.misc import get_device_count, is_env_enabled
    from .extras.mvp_policy import (
        read_mvp_train_request,
        run_mvp_dry_run,
        validate_mvp_runtime_environment,
        validate_mvp_train_config,
    )

    welcome = f"Multimodal Fine-tuning Platform CLI {VERSION}"
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "train" and len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        run_mvp_dry_run(sys.argv[2:])
        return

    if command == "train":
        validate_mvp_runtime_environment()
        validate_mvp_train_config(read_mvp_train_request(sys.argv[1:]))
        device_count = get_device_count()
        if is_env_enabled("FORCE_TORCHRUN") or device_count > 1:
            _launch_single_machine_workers(device_count)

        from .train.tuner import run_exp

        run_exp()
    elif command == "api":
        from .api.mvp_runtime import run_mvp_api

        run_mvp_api()
    elif command == "bootstrap":
        from .api.mvp_runtime import run_mvp_bootstrap

        run_mvp_bootstrap()
    elif command == "backup":
        from .mvp.operations import run_backup_plan_cli

        run_backup_plan_cli(sys.argv[1:])
    elif command == "restore":
        from .mvp.operations import run_restore_plan_cli

        run_restore_plan_cli(sys.argv[1:])
    elif command == "webui":
        from .webui.interface import run_web_ui

        run_web_ui()
    elif command == "env":
        print_env()
    elif command == "version":
        print(welcome)
    elif command == "help":
        print(USAGE)
    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from sft_platform.train.tuner import run_exp

    run_exp()
