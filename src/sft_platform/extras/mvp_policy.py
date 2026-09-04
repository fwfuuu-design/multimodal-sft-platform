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

"""Product-level policy for the multimodal fine-tuning MVP.

This module deliberately uses only the Python standard library. It is loaded
before training libraries so unsupported requests fail without touching a
model, a dataset, a GPU, or a distributed runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


_SUPPORTED_STAGE = "sft"
_SUPPORTED_FINETUNING_TYPE = "lora"
_SUPPORTED_QUANTIZATION_METHOD = "bnb"
_SUPPORTED_QUANTIZATION_BIT = 4

_UNSUPPORTED_BOOLEAN_FIELDS = {
    "enable_liger_kernel": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "fp8": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "fp8_enable_fsdp_float8_all_gather": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "pissa_convert": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "push_to_hub": "MVP_EXPORT_MODE_NOT_SUPPORTED",
    "pissa_init": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "shift_attn": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "train_from_scratch": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_adam_mini": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_apollo": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_asft_loss": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_badam": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_dft_loss": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_dora": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_eaft_loss": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_galore": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_hyper_parallel": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "use_kt": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "use_llama_pro": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_mca": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "use_megatron_bridge": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "use_muon": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_rslora": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_unsloth": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_unsloth_gc": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "use_audio_in_video": "MVP_VIDEO_TRAINING_NOT_SUPPORTED",
}

_UNSUPPORTED_VALUE_FIELDS = {
    "deepspeed": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "fsdp": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "fsdp_config": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "hub_model_id": "MVP_EXPORT_MODE_NOT_SUPPORTED",
    "hyper_parallel_args": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "loraplus_lr_ratio": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "master_addr": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "master_port": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "mixture_of_depths": "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED",
    "ray_init_kwargs": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "ray_num_workers": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
}

_UNSUPPORTED_ENV_FLAGS = {
    "USE_KT": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "USE_MCA": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "USE_MEGATRON_BRIDGE": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "USE_RAY": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
    "USE_V1": "MVP_DISTRIBUTED_BACKEND_NOT_SUPPORTED",
}

_FALSE_STRINGS = {"", "0", "false", "no", "none", "null", "off"}
_EMPTY_VALUES = (None, "", [], {}, ())


class MvpPolicyError(ValueError):
    """Stable, structured rejection raised at every formal training entry."""

    def __init__(self, code: str, field: str, value: Any, message: str):
        self.code = code
        self.field = field
        self.value = value
        super().__init__(f"{code}: {message} (field={field!r}, value={value!r})")


def _is_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


def _is_empty(value: Any) -> bool:
    return value in _EMPTY_VALUES or (isinstance(value, str) and value.strip().lower() in _FALSE_STRINGS)


def _normalize_key(key: str) -> str:
    return key.lstrip("-").replace("-", "_")


def _cli_args_to_mapping(args: Sequence[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            if "=" in token:
                key, value = token.split("=", 1)
                config[_normalize_key(key)] = value
            index += 1
            continue

        option = token[2:]
        if "=" in option:
            key, value = option.split("=", 1)
        elif index + 1 < len(args) and not args[index + 1].startswith("--"):
            key, value = option, args[index + 1]
            index += 1
        else:
            key, value = option, True

        config[_normalize_key(key)] = value
        index += 1

    return config


def _as_mapping(args: Mapping[str, Any] | Sequence[str]) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return {_normalize_key(str(key)): value for key, value in args.items()}
    if isinstance(args, (str, bytes)):
        raise TypeError("MVP training arguments must be a mapping or an argv sequence.")
    return _cli_args_to_mapping(args)


def validate_mvp_runtime_environment(environ: Mapping[str, str] | None = None) -> None:
    """Reject multi-node, elastic, Ray, Megatron, and alternate trainer runtimes."""
    environ = os.environ if environ is None else environ
    for field, code in _UNSUPPORTED_ENV_FLAGS.items():
        if _is_enabled(environ.get(field)):
            raise MvpPolicyError(code, field, environ.get(field), f"{field} is outside the MVP training runtime")

    nnodes = environ.get("NNODES", "1")
    try:
        node_count = int(nnodes)
    except (TypeError, ValueError) as err:
        raise MvpPolicyError(
            "MVP_INVALID_RESOURCE_REQUEST", "NNODES", nnodes, "NNODES must be the integer 1"
        ) from err

    if node_count != 1:
        raise MvpPolicyError(
            "MVP_MULTI_NODE_NOT_SUPPORTED", "NNODES", nnodes, "only single-machine training is supported"
        )

    node_rank = environ.get("NODE_RANK", "0")
    if str(node_rank) != "0":
        raise MvpPolicyError(
            "MVP_MULTI_NODE_NOT_SUPPORTED", "NODE_RANK", node_rank, "single-machine training requires node rank 0"
        )

    for field in ("RDZV_ID", "MIN_NNODES", "MAX_NNODES"):
        if not _is_empty(environ.get(field)):
            raise MvpPolicyError(
                "MVP_ELASTIC_TRAINING_NOT_SUPPORTED",
                field,
                environ.get(field),
                "elastic or rendezvous training is outside the MVP",
            )


def validate_mvp_train_config(args: Mapping[str, Any] | Sequence[str]) -> dict[str, Any]:
    """Validate and return a normalized copy of an MVP train configuration."""
    config = _as_mapping(args)

    stage = str(config.get("stage", _SUPPORTED_STAGE)).lower()
    if stage != _SUPPORTED_STAGE:
        raise MvpPolicyError(
            "MVP_TRAINING_STAGE_NOT_SUPPORTED", "stage", stage, "only supervised fine-tuning (sft) is supported"
        )

    finetuning_type = str(config.get("finetuning_type", _SUPPORTED_FINETUNING_TYPE)).lower()
    if finetuning_type != _SUPPORTED_FINETUNING_TYPE:
        raise MvpPolicyError(
            "MVP_FINETUNING_METHOD_NOT_SUPPORTED",
            "finetuning_type",
            finetuning_type,
            "only LoRA and 4-bit QLoRA are supported",
        )

    quantization_bit = config.get("quantization_bit")
    if isinstance(quantization_bit, str) and quantization_bit.strip().lower() in _FALSE_STRINGS:
        quantization_bit = None
    if quantization_bit is not None:
        try:
            quantization_bit = int(quantization_bit)
        except (TypeError, ValueError) as err:
            raise MvpPolicyError(
                "MVP_QUANTIZATION_NOT_SUPPORTED",
                "quantization_bit",
                quantization_bit,
                "QLoRA requires 4-bit quantization",
            ) from err

        if quantization_bit != _SUPPORTED_QUANTIZATION_BIT:
            raise MvpPolicyError(
                "MVP_QUANTIZATION_NOT_SUPPORTED",
                "quantization_bit",
                quantization_bit,
                "QLoRA requires 4-bit quantization",
            )

        quantization_method = str(config.get("quantization_method", _SUPPORTED_QUANTIZATION_METHOD)).lower()
        if quantization_method != _SUPPORTED_QUANTIZATION_METHOD:
            raise MvpPolicyError(
                "MVP_QUANTIZATION_BACKEND_NOT_SUPPORTED",
                "quantization_method",
                quantization_method,
                "4-bit QLoRA uses the bitsandbytes backend",
            )

    elif "quantization_method" in config:
        quantization_method = str(config["quantization_method"]).lower()
        if quantization_method != _SUPPORTED_QUANTIZATION_METHOD:
            raise MvpPolicyError(
                "MVP_QUANTIZATION_BACKEND_NOT_SUPPORTED",
                "quantization_method",
                quantization_method,
                "only the bitsandbytes QLoRA backend is supported for training",
            )

    for field, code in _UNSUPPORTED_BOOLEAN_FIELDS.items():
        if field in config and _is_enabled(config[field]):
            raise MvpPolicyError(code, field, config[field], f"{field} is outside the MVP")

    for field, code in _UNSUPPORTED_VALUE_FIELDS.items():
        if field in config and not _is_empty(config[field]):
            raise MvpPolicyError(code, field, config[field], f"{field} is outside the MVP")

    optimizer = str(config.get("optim", "")).lower()
    if "muon" in optimizer:
        raise MvpPolicyError(
            "MVP_ADVANCED_ALGORITHM_NOT_SUPPORTED", "optim", optimizer, "Muon is outside the MVP"
        )

    normalized = deepcopy(config)
    normalized["stage"] = _SUPPORTED_STAGE
    normalized["finetuning_type"] = _SUPPORTED_FINETUNING_TYPE
    if quantization_bit is not None:
        normalized["quantization_bit"] = _SUPPORTED_QUANTIZATION_BIT
        normalized["quantization_method"] = _SUPPORTED_QUANTIZATION_METHOD
    return normalized


def reject_unsupported_training_media(videos: Sequence[Any], audios: Sequence[Any]) -> None:
    """Enforce the text/image-only contract at the SFT processor boundary."""
    if videos:
        raise MvpPolicyError(
            "MVP_VIDEO_TRAINING_NOT_SUPPORTED", "videos", len(videos), "video samples are outside the MVP"
        )
    if audios:
        raise MvpPolicyError(
            "MVP_AUDIO_TRAINING_NOT_SUPPORTED", "audios", len(audios), "audio samples are outside the MVP"
        )


def load_mvp_train_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON/YAML train request without importing the training stack."""
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as err:
            raise RuntimeError("PyYAML is required to read YAML dry-run configurations.") from err
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Dry-run configuration must be a .json, .yaml, or .yml file.")

    if not isinstance(payload, dict):
        raise ValueError("Dry-run configuration must contain a mapping at the document root.")
    return payload


def read_mvp_train_request(argv: Sequence[str]) -> dict[str, Any]:
    """Read a CLI train request early enough to reject it before torchrun."""
    if argv and not argv[0].startswith("--") and Path(argv[0]).suffix.lower() in {".json", ".yaml", ".yml"}:
        config = load_mvp_train_config(argv[0])
        config.update(_cli_args_to_mapping(argv[1:]))
        return config
    return _cli_args_to_mapping(argv)


def _validate_dry_run_requirements(config: Mapping[str, Any]) -> None:
    if not _is_enabled(config.get("do_train")):
        raise MvpPolicyError("MVP_DRY_RUN_INVALID", "do_train", config.get("do_train"), "do_train must be true")
    for field in ("model_name_or_path", "dataset", "output_dir"):
        if _is_empty(config.get(field)):
            raise MvpPolicyError("MVP_DRY_RUN_INVALID", field, config.get(field), f"{field} is required")


class FakeExecutor:
    """Deterministic no-model executor used by stage-2 contract tests and CLI dry-runs."""

    def __init__(self, node_type: str = "local", gpu_count: int = 1):
        if node_type not in {"local", "cloud"}:
            raise MvpPolicyError(
                "MVP_INVALID_RESOURCE_REQUEST", "node_type", node_type, "node type must be local or cloud"
            )
        if not isinstance(gpu_count, int) or gpu_count < 1:
            raise MvpPolicyError(
                "MVP_INVALID_RESOURCE_REQUEST", "gpu_count", gpu_count, "at least one GPU is required"
            )
        self.node_type = node_type
        self.gpu_count = gpu_count

    def execute(
        self,
        config: Mapping[str, Any],
        gpu_ids: Sequence[int] | None = None,
        attempt: int = 1,
        dataset_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = validate_mvp_train_config(config)
        _validate_dry_run_requirements(normalized)
        requested_gpu_ids = list(range(self.gpu_count)) if gpu_ids is None else list(gpu_ids)
        if not requested_gpu_ids or len(set(requested_gpu_ids)) != len(requested_gpu_ids):
            raise MvpPolicyError(
                "MVP_INVALID_RESOURCE_REQUEST", "gpu_ids", requested_gpu_ids, "GPU IDs must be non-empty and unique"
            )
        if min(requested_gpu_ids) < 0 or max(requested_gpu_ids) >= self.gpu_count:
            raise MvpPolicyError(
                "MVP_INVALID_RESOURCE_REQUEST", "gpu_ids", requested_gpu_ids, "requested GPU is unavailable"
            )

        fingerprint = hashlib.sha256(
            json.dumps(
                {"config": normalized, "dataset_snapshot_id": dataset_snapshot_id},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        output_dir = str(normalized["output_dir"]).rstrip("/")
        checkpoint_path = f"{output_dir}/checkpoint-fake-0001"
        adapter_path = f"{output_dir}/adapter_model.fake.safetensors"
        events = [
            {"sequence": 1, "state": "VALIDATING"},
            {"sequence": 2, "state": "QUEUED"},
            {"sequence": 3, "state": "STARTING"},
            {
                "sequence": 4,
                "state": "RUNNING",
                "metrics": {
                    "loss": 1.0,
                    "learning_rate": 0.0001,
                    "epoch": 1.0,
                    "progress": 0.5,
                    "gpu_memory_bytes": None,
                },
            },
            {"sequence": 5, "state": "SUCCEEDED"},
        ]
        return {
            "executor": "fake",
            "job_id": f"fake-job-{fingerprint}",
            "attempt": attempt,
            "state": "SUCCEEDED",
            "dataset_snapshot_id": dataset_snapshot_id,
            "node": {
                "type": self.node_type,
                "gpu_count": self.gpu_count,
                "allocated_gpu_ids": requested_gpu_ids,
            },
            "events": events,
            "checkpoint": {
                "path": checkpoint_path,
                "step": 1,
                "config_fingerprint": fingerprint,
                "dataset_snapshot_id": dataset_snapshot_id,
                "restorable": True,
            },
            "artifacts": [
                {"kind": "adapter", "path": adapter_path, "fake": True},
                {"kind": "checkpoint", "path": checkpoint_path, "fake": True},
                {"kind": "log", "path": f"{output_dir}/training.fake.log", "fake": True},
            ],
        }

    def resume(
        self,
        config: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        gpu_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        if not checkpoint.get("restorable") or _is_empty(checkpoint.get("path")):
            raise MvpPolicyError(
                "MVP_CHECKPOINT_NOT_RESTORABLE", "checkpoint", checkpoint, "a restorable fake checkpoint is required"
            )
        resumed = deepcopy(dict(config))
        resumed["resume_from_checkpoint"] = checkpoint["path"]
        return self.execute(
            resumed,
            gpu_ids=gpu_ids,
            attempt=2,
            dataset_snapshot_id=checkpoint.get("dataset_snapshot_id"),
        )


def build_mvp_dry_run(config: Mapping[str, Any], executor: FakeExecutor | None = None) -> dict[str, Any]:
    """Build controlled YAML/argv previews and execute the deterministic fake flow."""
    normalized = validate_mvp_train_config(config)
    _validate_dry_run_requirements(normalized)
    fake_executor = executor or FakeExecutor()
    return {
        "dry_run": True,
        "mode": "qlora" if normalized.get("quantization_bit") == 4 else "lora",
        # JSON is a valid YAML 1.2 document and keeps this path deterministic.
        "yaml": json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "argv": ["sft-train", "train", "<controlled-job.yaml>"],
        "execution": fake_executor.execute(normalized),
    }


def run_mvp_dry_run(argv: Sequence[str]) -> dict[str, Any]:
    """CLI implementation for ``sft-train train --dry-run CONFIG``."""
    if len(argv) != 1:
        raise ValueError("Usage: sft-train train --dry-run <config.json|config.yaml>")
    report = build_mvp_dry_run(load_mvp_train_config(argv[0]))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report
