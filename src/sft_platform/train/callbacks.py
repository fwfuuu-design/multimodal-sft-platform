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

"""Callbacks required by the SFT workflow and local WebUI compatibility path."""

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import transformers
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG
from ..extras.misc import is_env_enabled


if TYPE_CHECKING:
    from transformers import ProcessorMixin, TrainerControl, TrainerState, TrainingArguments


logger = logging.get_logger(__name__)


class SaveProcessorCallback(TrainerCallback):
    r"""Save the multimodal processor beside each checkpoint and final Adapter."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class LogCallback(TrainerCallback):
    r"""Write stable JSONL progress records for SFT train/evaluate/predict."""

    def __init__(self) -> None:
        self.start_time = 0.0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.aborted = False
        self.do_train = False
        self.webui_mode = is_env_enabled("SFT_PLATFORM_WEBUI_ENABLED")
        if self.webui_mode:
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("SFT_PLATFORM_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        elapsed = time.time() - self.start_time
        average = elapsed / cur_steps if cur_steps else 0
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed)))
        self.remaining_time = str(timedelta(seconds=int((self.max_steps - cur_steps) * average)))

    @staticmethod
    def _write_log(output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as file:
            file.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        log_path = os.path.join(args.output_dir, TRAINER_LOG)
        if args.should_save and os.path.exists(log_path) and getattr(args, "overwrite_output_dir", False):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(log_path)

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return
        self._timing(state.global_step)
        current = state.log_history[-1]
        logs = {
            "current_steps": self.cur_steps,
            "total_steps": self.max_steps,
            "loss": current.get("loss"),
            "eval_loss": current.get("eval_loss"),
            "predict_loss": current.get("predict_loss"),
            "accuracy": current.get("eval_accuracy"),
            "lr": current.get("learning_rate"),
            "epoch": current.get("epoch"),
            "percentage": round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps else 100,
            "elapsed_time": self.elapsed_time,
            "remaining_time": self.remaining_time,
        }
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / max(time.time() - self.start_time, 1e-6), 2)
            logs["total_tokens"] = state.num_input_tokens_seen
        logs = {key: value for key, value in logs.items() if value is not None}
        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        if self.do_train:
            return
        if self.aborted:
            sys.exit(0)
        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(args.output_dir)
            self._timing(self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = {
                    "current_steps": self.cur_steps,
                    "total_steps": self.max_steps,
                    "percentage": round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps else 100,
                    "elapsed_time": self.elapsed_time,
                    "remaining_time": self.remaining_time,
                }
                self.thread_pool.submit(self._write_log, args.output_dir, logs)
