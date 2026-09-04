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

"""Persistent compute-node and training-job service for the MVP.

This module is intentionally independent from the model/training stack. A worker
receives a structured configuration and a fixed argv only; no API field is ever
interpreted as a shell command. The default test worker creates metadata-only
artifacts and never loads model weights.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..extras.mvp_policy import MvpPolicyError, validate_mvp_train_config


TERMINAL_STATES = {"SUCCEEDED", "FAILED", "STOPPED"}
ACTIVE_STATES = {"STARTING", "RUNNING", "STOPPING"}
LOCKED_CODE_VERSION = "4451765a6b04ff08a6c5650f5953513608ae9e64"
ERROR_CATEGORIES = {
    "DATA",
    "DISK",
    "ENVIRONMENT",
    "GPU_MEMORY",
    "MODEL",
    "NETWORK",
    "NODE_DISCONNECTED",
    "PERMISSION",
    "RESOURCE",
    "UNKNOWN",
}
STATE_TRANSITIONS = {
    "DRAFT": {"VALIDATING"},
    "VALIDATING": {"QUEUED", "FAILED"},
    "QUEUED": {"STARTING", "STOPPING", "FAILED"},
    "STARTING": {"RUNNING", "STOPPING", "FAILED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "STOPPING"},
    "STOPPING": {"STOPPED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
    "STOPPED": set(),
}

# Product-supported SFT knobs. Managed references and output paths are injected
# by the service and therefore deliberately absent from this set.
TRAINING_PARAMETER_ALLOWLIST = {
    "bf16",
    "cutoff_len",
    "do_train",
    "double_quantization",
    "eval_strategy",
    "finetuning_type",
    "flash_attn",
    "fp16",
    "freeze_language_model",
    "freeze_multi_modal_projector",
    "freeze_vision_tower",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "image_max_pixels",
    "image_min_pixels",
    "learning_rate",
    "logging_steps",
    "lora_rank",
    "lora_target",
    "lr_scheduler_type",
    "max_grad_norm",
    "max_samples",
    "neat_packing",
    "num_train_epochs",
    "packing",
    "per_device_train_batch_size",
    "quantization_bit",
    "quantization_method",
    "save_steps",
    "stage",
    "template",
    "val_size",
    "warmup_ratio",
}
MANAGED_PARAMETER_NAMES = {"dataset", "model_name_or_path", "output_dir", "resume_from_checkpoint"}
METRIC_ALLOWLIST = {
    "epoch",
    "estimated_remaining_seconds",
    "gpu_memory_bytes",
    "learning_rate",
    "loss",
    "progress",
}
_SECRET_KEY = re.compile(r"(authorization|cookie|password|secret|token)$", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(?i)(bearer\s+\S+|hf_[A-Za-z0-9_-]+|ms_token_[A-Za-z0-9_-]+)")


class JobServiceError(ValueError):
    def __init__(self, code: str, field: str, message: str, status_code: int = 422):
        self.code = code
        self.field = field
        self.status_code = status_code
        super().__init__(f"{code}: {message} (field={field!r})")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": str(self)}


@dataclass(frozen=True)
class ComputeNodeRegistration:
    name: str
    node_type: str
    endpoint: str
    model_root: str
    data_root: str
    output_root: str
    labels: Mapping[str, str]


@dataclass(frozen=True)
class GpuReport:
    id: int
    model: str
    memory_total_bytes: int
    memory_free_bytes: int
    utilization: float = 0.0


@dataclass(frozen=True)
class JobSubmission:
    name: str
    model_id: str
    dataset_snapshot_id: str
    training_parameters: Mapping[str, Any]
    gpu_count: int = 1
    node_id: str | None = None
    minimum_disk_bytes: int = 0


class SystemClock:
    def now(self) -> float:
        return datetime.now(UTC).timestamp()


class RandomIdGenerator:
    def __call__(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode(value: str | None, default: Any) -> Any:
    return deepcopy(default) if value is None else json.loads(value)


def _redact(value: str) -> str:
    return _SECRET_VALUE.sub("[REDACTED]", value)


def _contains_secret(value: Any, key: str = "") -> bool:
    if key and _SECRET_KEY.search(key) and not key.lower().endswith("credential_id"):
        return True
    if isinstance(value, Mapping):
        return any(_contains_secret(item, str(item_key)) for item_key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_secret(item) for item in value)
    return isinstance(value, str) and _SECRET_VALUE.search(value) is not None


def _controlled_root(raw_path: str, field: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise JobServiceError("NODE_ROOT_INVALID", field, "Worker roots must be absolute paths.")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise JobServiceError("NODE_ROOT_NOT_FOUND", field, "Worker root does not exist.")
    return str(resolved)


def _controlled_child(raw_path: str, root: str, field: str) -> str:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_relative_to(Path(root)):
        raise JobServiceError("PATH_OUTSIDE_CONTROLLED_ROOT", field, "Path is outside the node task root.")
    return str(path)


class TrainingJobService:
    """SQLite-backed node registry, queue, scheduler, and Agent protocol."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Any | None = None,
        id_generator: Callable[[str], str] | None = None,
        lease_ttl_seconds: int = 60,
        heartbeat_timeout_seconds: int = 90,
        max_log_chunk_bytes: int = 64 * 1024,
        model_service: Any | None = None,
        dataset_service: Any | None = None,
        code_version: str = LOCKED_CODE_VERSION,
    ):
        self.database_path = str(Path(database_path).expanduser().resolve())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or RandomIdGenerator()
        self.lease_ttl_seconds = lease_ttl_seconds
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.max_log_chunk_bytes = max_log_chunk_bytes
        self.model_service = model_service
        self.dataset_service = dataset_service
        self.code_version = code_version
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY, name TEXT NOT NULL, node_type TEXT NOT NULL,
                    endpoint TEXT NOT NULL, status TEXT NOT NULL, labels_json TEXT NOT NULL,
                    model_root TEXT NOT NULL, data_root TEXT NOT NULL, output_root TEXT NOT NULL,
                    driver_version TEXT, disk_free_bytes INTEGER, system_memory_free_bytes INTEGER,
                    last_heartbeat REAL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gpus (
                    node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                    gpu_id INTEGER NOT NULL, model TEXT NOT NULL, memory_total_bytes INTEGER NOT NULL,
                    memory_free_bytes INTEGER NOT NULL, utilization REAL NOT NULL,
                    PRIMARY KEY (node_id, gpu_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
                    node_id TEXT REFERENCES nodes(node_id), gpu_count INTEGER NOT NULL,
                    minimum_disk_bytes INTEGER NOT NULL, model_id TEXT NOT NULL,
                    dataset_snapshot_id TEXT NOT NULL, config_json TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL, code_version TEXT NOT NULL, output_dir TEXT,
                    parent_job_id TEXT, parent_checkpoint_id TEXT,
                    error_category TEXT, error_code TEXT, error_message TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    number INTEGER NOT NULL, state TEXT NOT NULL, node_id TEXT REFERENCES nodes(node_id),
                    allocated_gpus_json TEXT NOT NULL, started_at REAL, ended_at REAL,
                    UNIQUE (job_id, number)
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    attempt_id TEXT, event_type TEXT NOT NULL, state TEXT,
                    payload_json TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS logs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL, stream TEXT NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
                    kind TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER, sha256 TEXT,
                    metadata_json TEXT NOT NULL, restorable INTEGER NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                    node_id TEXT NOT NULL, gpu_id INTEGER NOT NULL, job_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL, expires_at REAL NOT NULL,
                    PRIMARY KEY (node_id, gpu_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(state, created_at);
                CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id, sequence);
                CREATE INDEX IF NOT EXISTS logs_job_idx ON logs(job_id, sequence);
                """
            )
            job_columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "code_version" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN code_version TEXT NOT NULL DEFAULT 'unknown'")
            node_columns = {row["name"] for row in connection.execute("PRAGMA table_info(nodes)")}
            if "system_memory_free_bytes" not in node_columns:
                connection.execute("ALTER TABLE nodes ADD COLUMN system_memory_free_bytes INTEGER")

    def register_node(self, registration: ComputeNodeRegistration) -> dict[str, Any]:
        if registration.node_type not in {"local", "cloud"}:
            raise JobServiceError("NODE_TYPE_NOT_SUPPORTED", "node_type", "Node type must be local or cloud.")
        if not registration.name.strip():
            raise JobServiceError("NODE_NAME_REQUIRED", "name", "Node name is required.")
        if not registration.endpoint.startswith(("http://", "https://")):
            raise JobServiceError("NODE_ENDPOINT_INVALID", "endpoint", "Agent endpoint must use HTTP or HTTPS.")
        if _contains_secret(registration.endpoint) or _contains_secret(registration.labels):
            raise JobServiceError("PLAINTEXT_CREDENTIAL_NOT_ALLOWED", "node", "Node metadata contains a secret.")
        roots = {
            "model_root": _controlled_root(registration.model_root, "model_root"),
            "data_root": _controlled_root(registration.data_root, "data_root"),
            "output_root": _controlled_root(registration.output_root, "output_root"),
        }
        node_id = self.id_generator("node")
        now = self.clock.now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO nodes (node_id, name, node_type, endpoint, status, labels_json, model_root, data_root, "
                "output_root, driver_version, disk_free_bytes, system_memory_free_bytes, last_heartbeat, created_at) "
                "VALUES (?, ?, ?, ?, 'OFFLINE', ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)",
                (
                    node_id,
                    registration.name.strip(),
                    registration.node_type,
                    registration.endpoint,
                    _json(dict(registration.labels)),
                    roots["model_root"],
                    roots["data_root"],
                    roots["output_root"],
                    now,
                ),
            )
        return self.get_node(node_id)

    def heartbeat(
        self,
        node_id: str,
        gpus: Sequence[GpuReport],
        *,
        disk_free_bytes: int,
        driver_version: str,
        system_memory_free_bytes: int | None = None,
        active_attempt_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        if disk_free_bytes < 0:
            raise JobServiceError("NODE_DISK_REPORT_INVALID", "disk_free_bytes", "Disk bytes cannot be negative.")
        if system_memory_free_bytes is not None and system_memory_free_bytes < 0:
            raise JobServiceError(
                "NODE_MEMORY_REPORT_INVALID", "system_memory_free_bytes", "System memory bytes cannot be negative."
            )
        gpu_ids = [gpu.id for gpu in gpus]
        if len(gpu_ids) != len(set(gpu_ids)) or any(gpu_id < 0 for gpu_id in gpu_ids):
            raise JobServiceError("NODE_GPU_REPORT_INVALID", "gpus", "GPU IDs must be unique non-negative integers.")
        for gpu in gpus:
            if (
                not gpu.model
                or gpu.memory_total_bytes < 1
                or gpu.memory_free_bytes < 0
                or gpu.memory_free_bytes > gpu.memory_total_bytes
                or not 0 <= gpu.utilization <= 1
            ):
                raise JobServiceError("NODE_GPU_REPORT_INVALID", "gpus", "GPU inventory contains invalid values.")

        now = self.clock.now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM nodes WHERE node_id = ?", (node_id,)).fetchone() is None:
                connection.rollback()
                raise JobServiceError("NODE_NOT_FOUND", "node_id", f"Unknown node: {node_id}", 404)
            connection.execute(
                "UPDATE nodes SET status = 'ONLINE', driver_version = ?, disk_free_bytes = ?, "
                "system_memory_free_bytes = ?, last_heartbeat = ? WHERE node_id = ?",
                (driver_version, disk_free_bytes, system_memory_free_bytes, now, node_id),
            )
            connection.execute("DELETE FROM gpus WHERE node_id = ?", (node_id,))
            connection.executemany(
                "INSERT INTO gpus VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        node_id,
                        gpu.id,
                        gpu.model,
                        gpu.memory_total_bytes,
                        gpu.memory_free_bytes,
                        gpu.utilization,
                    )
                    for gpu in gpus
                ],
            )
            active_ids = list(dict.fromkeys(active_attempt_ids))
            if active_ids:
                placeholders = ",".join("?" for _ in active_ids)
                connection.execute(
                    f"UPDATE leases SET expires_at = ? WHERE node_id = ? AND attempt_id IN ({placeholders})",
                    (now + self.lease_ttl_seconds, node_id, *active_ids),
                )
            connection.commit()
        node = self.get_node(node_id)
        node["commands"] = self._node_commands(node_id)
        return node

    def list_nodes(self) -> list[dict[str, Any]]:
        self.reconcile()
        with self._connect() as connection:
            rows = connection.execute("SELECT node_id FROM nodes ORDER BY created_at, node_id").fetchall()
        return [self.get_node(row["node_id"], reconcile=False) for row in rows]

    def get_node(self, node_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        if reconcile:
            self.reconcile()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            if row is None:
                raise JobServiceError("NODE_NOT_FOUND", "node_id", f"Unknown node: {node_id}", 404)
            gpu_rows = connection.execute("SELECT * FROM gpus WHERE node_id = ? ORDER BY gpu_id", (node_id,)).fetchall()
            leases = {
                lease["gpu_id"]: lease
                for lease in connection.execute(
                    "SELECT gpu_id, job_id, attempt_id, expires_at FROM leases WHERE node_id = ?", (node_id,)
                ).fetchall()
            }
        return {
            "id": row["node_id"],
            "name": row["name"],
            "type": row["node_type"],
            "endpoint": row["endpoint"],
            "status": row["status"],
            "labels": _decode(row["labels_json"], {}),
            "work_roots": {
                "models": row["model_root"],
                "datasets": row["data_root"],
                "outputs": row["output_root"],
            },
            "driver_version": row["driver_version"],
            "disk_free_bytes": row["disk_free_bytes"],
            "system_memory_free_bytes": row["system_memory_free_bytes"],
            "last_heartbeat": row["last_heartbeat"],
            "gpus": [
                {
                    "id": gpu["gpu_id"],
                    "model": gpu["model"],
                    "memory_total_bytes": gpu["memory_total_bytes"],
                    "memory_free_bytes": gpu["memory_free_bytes"],
                    "utilization": gpu["utilization"],
                    "lease": (
                        {
                            "job_id": leases[gpu["gpu_id"]]["job_id"],
                            "attempt_id": leases[gpu["gpu_id"]]["attempt_id"],
                            "expires_at": leases[gpu["gpu_id"]]["expires_at"],
                        }
                        if gpu["gpu_id"] in leases
                        else None
                    ),
                }
                for gpu in gpu_rows
            ],
        }

    def create_job(self, submission: JobSubmission, *, parent: Mapping[str, str] | None = None) -> dict[str, Any]:
        if not submission.name.strip():
            raise JobServiceError("JOB_NAME_REQUIRED", "name", "Job name is required.")
        if not submission.model_id.startswith("model-"):
            raise JobServiceError("MODEL_REFERENCE_INVALID", "model_id", "A registered model ID is required.")
        if not submission.dataset_snapshot_id.startswith("dataset-"):
            raise JobServiceError(
                "DATASET_REFERENCE_INVALID", "dataset_snapshot_id", "An immutable dataset snapshot ID is required."
            )
        dataset_records = None
        if self.model_service is not None:
            try:
                model = self.model_service.get(submission.model_id)
            except Exception as error:
                raise JobServiceError("MODEL_NOT_FOUND", "model_id", "Registered model does not exist.", 404) from error
            if not model.get("compatibility", {}).get("mvp_compatible"):
                raise JobServiceError("MODEL_NOT_MVP_COMPATIBLE", "model_id", "Model is not compatible with MVP SFT.")
        else:
            model = None
        if self.dataset_service is not None:
            try:
                dataset_records = self.dataset_service.get_snapshot(submission.dataset_snapshot_id)
            except Exception as error:
                raise JobServiceError(
                    "DATASET_SNAPSHOT_NOT_FOUND", "dataset_snapshot_id", "Dataset snapshot does not exist.", 404
                ) from error
        if model is not None and dataset_records is not None:
            required_mode = (
                "image_text_sft" if any(record.get("images") for record in dataset_records) else "text_sft"
            )
            if required_mode not in model.get("compatibility", {}).get("modes", []):
                raise JobServiceError(
                    "MODEL_DATASET_MODALITY_MISMATCH", "model_id", f"Model does not support {required_mode}."
                )
        if not isinstance(submission.gpu_count, int) or submission.gpu_count < 1:
            raise JobServiceError("RESOURCE_REQUEST_INVALID", "gpu_count", "At least one GPU is required.")
        if submission.minimum_disk_bytes < 0:
            raise JobServiceError("RESOURCE_REQUEST_INVALID", "minimum_disk_bytes", "Disk bytes cannot be negative.")
        if _contains_secret(submission.training_parameters):
            raise JobServiceError(
                "PLAINTEXT_CREDENTIAL_NOT_ALLOWED", "training_parameters", "Training parameters contain a secret."
            )
        managed = MANAGED_PARAMETER_NAMES.intersection(submission.training_parameters)
        if managed:
            raise JobServiceError(
                "MANAGED_PARAMETER_NOT_ALLOWED",
                sorted(managed)[0],
                "Model, dataset, output, and checkpoint paths are managed by the service.",
            )

        provisional = {
            **dict(submission.training_parameters),
            "model_name_or_path": submission.model_id,
            "dataset": submission.dataset_snapshot_id,
            "output_dir": "<managed-output-dir>",
            "do_train": True,
        }
        try:
            normalized = validate_mvp_train_config(provisional)
        except MvpPolicyError as error:
            raise JobServiceError(error.code, error.field, str(error)) from error
        unknown = set(submission.training_parameters).difference(TRAINING_PARAMETER_ALLOWLIST)
        if unknown:
            field = sorted(unknown)[0]
            raise JobServiceError("TRAINING_PARAMETER_NOT_ALLOWED", field, "Parameter is not in the SFT allowlist.")

        job_id = self.id_generator("job")
        attempt_id = self.id_generator("attempt")
        now = self.clock.now()
        output_dir = None
        if submission.node_id is not None:
            node = self.get_node(submission.node_id)
            if len(node["gpus"]) < submission.gpu_count:
                raise JobServiceError(
                    "GPU_RESOURCE_INSUFFICIENT", "gpu_count", "Selected node has fewer GPUs than requested.", 409
                )
            if node["disk_free_bytes"] is not None and node["disk_free_bytes"] < submission.minimum_disk_bytes:
                raise JobServiceError("DISK_INSUFFICIENT", "minimum_disk_bytes", "Selected node has insufficient disk.", 409)
            output_dir = str(Path(node["work_roots"]["outputs"]) / job_id)
        normalized["output_dir"] = output_dir or f"<node-output-root>/{job_id}"
        fingerprint = hashlib.sha256(_json(normalized).encode()).hexdigest()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if submission.node_id is not None and connection.execute(
                "SELECT 1 FROM nodes WHERE node_id = ?", (submission.node_id,)
            ).fetchone() is None:
                connection.rollback()
                raise JobServiceError("NODE_NOT_FOUND", "node_id", f"Unknown node: {submission.node_id}", 404)
            connection.execute(
                "INSERT INTO jobs (job_id, name, state, node_id, gpu_count, minimum_disk_bytes, model_id, "
                "dataset_snapshot_id, config_json, config_fingerprint, code_version, output_dir, parent_job_id, "
                "parent_checkpoint_id, error_category, error_code, error_message, created_at, updated_at) "
                "VALUES (?, ?, 'DRAFT', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)",
                (
                    job_id,
                    submission.name.strip(),
                    submission.node_id,
                    submission.gpu_count,
                    submission.minimum_disk_bytes,
                    submission.model_id,
                    submission.dataset_snapshot_id,
                    _json(normalized),
                    fingerprint,
                    self.code_version,
                    output_dir,
                    parent.get("job_id") if parent else None,
                    parent.get("checkpoint_id") if parent else None,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO attempts VALUES (?, ?, 1, 'DRAFT', NULL, '[]', NULL, NULL)",
                (attempt_id, job_id),
            )
            self._transition_tx(connection, job_id, "VALIDATING", attempt_id=attempt_id)
            self._transition_tx(connection, job_id, "QUEUED", attempt_id=attempt_id)
            connection.commit()
        return self.get_job(job_id)

    def list_jobs(self, *, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        self.reconcile()
        if limit < 1 or limit > 500:
            raise JobServiceError("QUERY_LIMIT_INVALID", "limit", "Limit must be 1..500.")
        with self._connect() as connection:
            if state is None:
                rows = connection.execute(
                    "SELECT job_id FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT job_id FROM jobs WHERE state = ? ORDER BY created_at DESC, job_id DESC LIMIT ?",
                    (state, limit),
                ).fetchall()
        return [self.get_job(row["job_id"], reconcile=False) for row in rows]

    def get_job(self, job_id: str, *, reconcile: bool = True) -> dict[str, Any]:
        if reconcile:
            self.reconcile()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobServiceError("JOB_NOT_FOUND", "job_id", f"Unknown job: {job_id}", 404)
            attempts = connection.execute(
                "SELECT * FROM attempts WHERE job_id = ? ORDER BY number", (job_id,)
            ).fetchall()
        return {
            "id": row["job_id"],
            "name": row["name"],
            "state": row["state"],
            "node_id": row["node_id"],
            "resources": {"gpu_count": row["gpu_count"], "minimum_disk_bytes": row["minimum_disk_bytes"]},
            "model_id": row["model_id"],
            "dataset_snapshot_id": row["dataset_snapshot_id"],
            "config_snapshot": _decode(row["config_json"], {}),
            "config_fingerprint": row["config_fingerprint"],
            "code_version": row["code_version"],
            "output_dir": row["output_dir"],
            "parent_job_id": row["parent_job_id"],
            "parent_checkpoint_id": row["parent_checkpoint_id"],
            "error": (
                {
                    "category": row["error_category"],
                    "code": row["error_code"],
                    "message": row["error_message"],
                }
                if row["error_code"]
                else None
            ),
            "attempts": [
                {
                    "id": attempt["attempt_id"],
                    "number": attempt["number"],
                    "state": attempt["state"],
                    "node_id": attempt["node_id"],
                    "allocated_gpu_ids": _decode(attempt["allocated_gpus_json"], []),
                    "started_at": attempt["started_at"],
                    "ended_at": attempt["ended_at"],
                }
                for attempt in attempts
            ],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "command_preview": ["sft-train", "train", "<controlled-job.yaml>"],
        }

    def poll_assignment(self, node_id: str) -> dict[str, Any] | None:
        now = self.clock.now()
        self.reconcile()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            node = connection.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            if node is None:
                connection.rollback()
                raise JobServiceError("NODE_NOT_FOUND", "node_id", f"Unknown node: {node_id}", 404)
            if node["status"] != "ONLINE":
                connection.rollback()
                raise JobServiceError("NODE_OFFLINE", "node_id", "Node must heartbeat before polling.", 409)

            jobs = connection.execute(
                "SELECT * FROM jobs WHERE state = 'QUEUED' AND (node_id IS NULL OR node_id = ?) "
                "ORDER BY created_at, job_id",
                (node_id,),
            ).fetchall()
            leased = {
                row["gpu_id"]
                for row in connection.execute("SELECT gpu_id FROM leases WHERE node_id = ?", (node_id,)).fetchall()
            }
            total_gpu_ids = [
                row["gpu_id"]
                for row in connection.execute("SELECT gpu_id FROM gpus WHERE node_id = ? ORDER BY gpu_id", (node_id,))
            ]
            available = [gpu_id for gpu_id in total_gpu_ids if gpu_id not in leased]
            selected = None
            for job in jobs:
                attempt = connection.execute(
                    "SELECT attempt_id FROM attempts WHERE job_id = ? ORDER BY number DESC LIMIT 1", (job["job_id"],)
                ).fetchone()
                if node["disk_free_bytes"] is not None and node["disk_free_bytes"] < job["minimum_disk_bytes"]:
                    if job["node_id"] == node_id:
                        self._fail_tx(
                            connection,
                            job["job_id"],
                            attempt["attempt_id"],
                            "DISK",
                            "DISK_INSUFFICIENT",
                            "Selected node no longer has enough free disk.",
                        )
                    continue
                if len(total_gpu_ids) < job["gpu_count"]:
                    if job["node_id"] == node_id:
                        self._fail_tx(
                            connection,
                            job["job_id"],
                            attempt["attempt_id"],
                            "RESOURCE",
                            "GPU_RESOURCE_INSUFFICIENT",
                            "Selected node no longer has enough GPUs.",
                        )
                    continue
                if len(available) >= job["gpu_count"]:
                    selected = job
                    break
            if selected is None:
                connection.commit()
                return None

            gpu_ids = available[: selected["gpu_count"]]
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE job_id = ? ORDER BY number DESC LIMIT 1", (selected["job_id"],)
            ).fetchone()
            output_dir = str(Path(node["output_root"]) / selected["job_id"])
            config = _decode(selected["config_json"], {})
            config["output_dir"] = output_dir
            if selected["parent_checkpoint_id"]:
                checkpoint = connection.execute(
                    "SELECT path FROM artifacts WHERE artifact_id = ? AND restorable = 1",
                    (selected["parent_checkpoint_id"],),
                ).fetchone()
                if checkpoint is None:
                    self._fail_tx(
                        connection,
                        selected["job_id"],
                        attempt["attempt_id"],
                        "DATA",
                        "CHECKPOINT_NOT_RESTORABLE",
                        "Parent checkpoint is missing or not restorable.",
                    )
                    connection.commit()
                    return None
                config["resume_from_checkpoint"] = checkpoint["path"]
            connection.execute(
                "UPDATE jobs SET node_id = ?, output_dir = ?, config_json = ?, updated_at = ? WHERE job_id = ?",
                (node_id, output_dir, _json(config), now, selected["job_id"]),
            )
            connection.execute(
                "UPDATE attempts SET node_id = ?, allocated_gpus_json = ? WHERE attempt_id = ?",
                (node_id, _json(gpu_ids), attempt["attempt_id"]),
            )
            connection.executemany(
                "INSERT INTO leases VALUES (?, ?, ?, ?, ?)",
                [
                    (node_id, gpu_id, selected["job_id"], attempt["attempt_id"], now + self.lease_ttl_seconds)
                    for gpu_id in gpu_ids
                ],
            )
            self._transition_tx(connection, selected["job_id"], "STARTING", attempt_id=attempt["attempt_id"])
            connection.commit()
        return {
            "job_id": selected["job_id"],
            "attempt_id": attempt["attempt_id"],
            "gpu_ids": gpu_ids,
            "config": config,
            "argv": ["sft-train", "train", "<controlled-job.yaml>"],
            "lease_expires_at": now + self.lease_ttl_seconds,
        }

    def report_state(
        self,
        node_id: str,
        attempt_id: str,
        state: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        error_category: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        state = state.upper()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if attempt is None:
                connection.rollback()
                raise JobServiceError("ATTEMPT_NOT_FOUND", "attempt_id", "Unknown attempt.", 404)
            if attempt["node_id"] != node_id:
                connection.rollback()
                raise JobServiceError("ATTEMPT_NODE_MISMATCH", "node_id", "Attempt belongs to another node.", 409)
            job = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (attempt["job_id"],)).fetchone()
            if state == "FAILED":
                category = error_category or "UNKNOWN"
                if category not in ERROR_CATEGORIES:
                    connection.rollback()
                    raise JobServiceError("ERROR_CATEGORY_INVALID", "error_category", "Unknown error category.")
                self._fail_tx(
                    connection,
                    job["job_id"],
                    attempt_id,
                    category,
                    error_code or "EXECUTOR_FAILED",
                    _redact(error_message or "Worker reported a failure."),
                )
            else:
                if metrics is not None:
                    unknown_metrics = set(metrics).difference(METRIC_ALLOWLIST)
                    invalid_values = [
                        key
                        for key, value in metrics.items()
                        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
                    ]
                    if unknown_metrics or invalid_values or _contains_secret(metrics):
                        connection.rollback()
                        field = sorted(unknown_metrics or set(invalid_values) or {"metrics"})[0]
                        raise JobServiceError("METRICS_INVALID", field, "Worker metrics contain an invalid field or value.")
                payload = {"metrics": dict(metrics)} if metrics is not None else {}
                self._transition_tx(connection, job["job_id"], state, attempt_id=attempt_id, payload=payload)
                if state in TERMINAL_STATES:
                    connection.execute("DELETE FROM leases WHERE attempt_id = ?", (attempt_id,))
            connection.commit()
        return self.get_job(attempt["job_id"])

    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if job is None:
                connection.rollback()
                raise JobServiceError("JOB_NOT_FOUND", "job_id", f"Unknown job: {job_id}", 404)
            if job["state"] in TERMINAL_STATES:
                connection.rollback()
                raise JobServiceError("JOB_NOT_STOPPABLE", "state", "Terminal job cannot be stopped.", 409)
            attempt = connection.execute(
                "SELECT attempt_id FROM attempts WHERE job_id = ? ORDER BY number DESC LIMIT 1", (job_id,)
            ).fetchone()
            self._transition_tx(connection, job_id, "STOPPING", attempt_id=attempt["attempt_id"])
            if job["state"] == "QUEUED":
                self._transition_tx(connection, job_id, "STOPPED", attempt_id=attempt["attempt_id"])
                connection.execute("DELETE FROM leases WHERE attempt_id = ?", (attempt["attempt_id"],))
            connection.commit()
        return self.get_job(job_id)

    def resume_job(self, job_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job["state"] not in TERMINAL_STATES:
            raise JobServiceError("JOB_NOT_RESUMABLE", "state", "Only terminal jobs can be resumed.", 409)
        artifacts = self.list_artifacts(job_id)
        checkpoints = [artifact for artifact in artifacts if artifact["kind"] == "checkpoint" and artifact["restorable"]]
        if checkpoint_id is None and checkpoints:
            checkpoint_id = checkpoints[-1]["id"]
        checkpoint = next((item for item in checkpoints if item["id"] == checkpoint_id), None)
        if checkpoint is None:
            raise JobServiceError(
                "CHECKPOINT_NOT_RESTORABLE", "checkpoint_id", "A restorable checkpoint from this job is required.", 409
            )
        params = {
            key: value
            for key, value in job["config_snapshot"].items()
            if key in TRAINING_PARAMETER_ALLOWLIST and key != "do_train"
        }
        return self.create_job(
            JobSubmission(
                name=f"{job['name']} (resume)",
                model_id=job["model_id"],
                dataset_snapshot_id=job["dataset_snapshot_id"],
                training_parameters=params,
                gpu_count=job["resources"]["gpu_count"],
                node_id=job["node_id"],
                minimum_disk_bytes=job["resources"]["minimum_disk_bytes"],
            ),
            parent={"job_id": job_id, "checkpoint_id": checkpoint["id"]},
        )

    def append_logs(self, node_id: str, attempt_id: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        encoded_size = len(_json(list(entries)).encode())
        if encoded_size > self.max_log_chunk_bytes:
            raise JobServiceError("LOG_RATE_LIMIT_EXCEEDED", "entries", "Log chunk exceeds the configured limit.", 429)
        if not entries:
            return {"accepted": 0}
        with self._connect() as connection:
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if attempt is None:
                raise JobServiceError("ATTEMPT_NOT_FOUND", "attempt_id", "Unknown attempt.", 404)
            if attempt["node_id"] != node_id:
                raise JobServiceError("ATTEMPT_NODE_MISMATCH", "node_id", "Attempt belongs to another node.", 409)
            now = self.clock.now()
            clean_entries = []
            for entry in entries:
                stream = str(entry.get("stream", "stdout"))
                message = entry.get("message")
                if stream not in {"stdout", "stderr", "system"} or not isinstance(message, str):
                    raise JobServiceError("LOG_ENTRY_INVALID", "entries", "Log stream or message is invalid.")
                clean_entries.append((attempt["job_id"], attempt_id, stream, _redact(message), now))
            connection.executemany(
                "INSERT INTO logs(job_id, attempt_id, stream, message, created_at) VALUES (?, ?, ?, ?, ?)",
                clean_entries,
            )
        return {"accepted": len(entries)}

    def get_logs(self, job_id: str, *, after_sequence: int = 0, limit: int = 1000) -> dict[str, Any]:
        self._ensure_job(job_id)
        if limit < 1 or limit > 1000:
            raise JobServiceError("QUERY_LIMIT_INVALID", "limit", "Log limit must be 1..1000.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM logs WHERE job_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (job_id, after_sequence, limit),
            ).fetchall()
        items = [dict(row) for row in rows]
        return {"items": items, "next_sequence": items[-1]["sequence"] if items else after_sequence}

    def add_artifacts(
        self, node_id: str, attempt_id: str, artifacts: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if attempt is None:
                connection.rollback()
                raise JobServiceError("ATTEMPT_NOT_FOUND", "attempt_id", "Unknown attempt.", 404)
            if attempt["node_id"] != node_id:
                connection.rollback()
                raise JobServiceError("ATTEMPT_NODE_MISMATCH", "node_id", "Attempt belongs to another node.", 409)
            job = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (attempt["job_id"],)).fetchone()
            now = self.clock.now()
            for artifact in artifacts:
                kind = str(artifact.get("kind", ""))
                if kind not in {"adapter", "checkpoint", "config", "log", "metrics", "other"}:
                    connection.rollback()
                    raise JobServiceError("ARTIFACT_KIND_INVALID", "kind", "Unsupported artifact kind.")
                path = _controlled_child(str(artifact.get("path", "")), job["output_dir"], "path")
                size = artifact.get("bytes")
                if size is not None and (not isinstance(size, int) or size < 0):
                    connection.rollback()
                    raise JobServiceError("ARTIFACT_SIZE_INVALID", "bytes", "Artifact bytes must be non-negative.")
                metadata = dict(artifact.get("metadata", {}))
                if _contains_secret(metadata):
                    connection.rollback()
                    raise JobServiceError("PLAINTEXT_CREDENTIAL_NOT_ALLOWED", "metadata", "Artifact metadata has a secret.")
                artifact_id = self.id_generator("artifact")
                connection.execute(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        artifact_id,
                        job["job_id"],
                        attempt_id,
                        kind,
                        path,
                        size,
                        artifact.get("sha256"),
                        _json(metadata),
                        int(bool(artifact.get("restorable", False) and kind == "checkpoint")),
                        now,
                    ),
                )
                self._event_tx(
                    connection,
                    job["job_id"],
                    attempt_id,
                    "ARTIFACT_CREATED",
                    job["state"],
                    {"artifact_id": artifact_id, "kind": kind},
                )
            connection.commit()
        return self.list_artifacts(attempt["job_id"])

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT job_id FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise JobServiceError("ARTIFACT_NOT_FOUND", "artifact_id", "Artifact does not exist.", 404)
        return next(item for item in self.list_artifacts(row["job_id"]) if item["id"] == artifact_id)

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        self._ensure_job(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at, artifact_id", (job_id,)
            ).fetchall()
        return [
            {
                "id": row["artifact_id"],
                "job_id": row["job_id"],
                "attempt_id": row["attempt_id"],
                "kind": row["kind"],
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "metadata": _decode(row["metadata_json"], {}),
                "restorable": bool(row["restorable"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_events(self, job_id: str, *, after_sequence: int = 0, limit: int = 1000) -> dict[str, Any]:
        self._ensure_job(job_id)
        if limit < 1 or limit > 1000:
            raise JobServiceError("QUERY_LIMIT_INVALID", "limit", "Event limit must be 1..1000.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE job_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (job_id, after_sequence, limit),
            ).fetchall()
        items = [
            {
                "sequence": row["sequence"],
                "attempt_id": row["attempt_id"],
                "type": row["event_type"],
                "state": row["state"],
                "payload": _decode(row["payload_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {"items": items, "next_sequence": items[-1]["sequence"] if items else after_sequence}

    def reconcile(self) -> None:
        """Converge stale nodes/leases without ever reallocating an active attempt."""
        now = self.clock.now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stale_nodes = connection.execute(
                "SELECT node_id FROM nodes WHERE status = 'ONLINE' AND last_heartbeat < ?",
                (now - self.heartbeat_timeout_seconds,),
            ).fetchall()
            connection.executemany(
                "UPDATE nodes SET status = 'OFFLINE' WHERE node_id = ?", [(row["node_id"],) for row in stale_nodes]
            )
            expired_attempts = connection.execute(
                "SELECT DISTINCT attempt_id, job_id FROM leases WHERE expires_at <= ?", (now,)
            ).fetchall()
            for row in expired_attempts:
                job = connection.execute("SELECT state FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
                if job is not None and job["state"] in ACTIVE_STATES:
                    self._fail_tx(
                        connection,
                        row["job_id"],
                        row["attempt_id"],
                        "NODE_DISCONNECTED",
                        "GPU_LEASE_EXPIRED",
                        "Worker heartbeat expired; the attempt was not reassigned.",
                    )
            connection.execute("DELETE FROM leases WHERE expires_at <= ?", (now,))
            connection.commit()

    def _node_commands(self, node_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT a.attempt_id, a.job_id FROM attempts a JOIN jobs j ON j.job_id = a.job_id "
                "WHERE a.node_id = ? AND j.state = 'STOPPING'",
                (node_id,),
            ).fetchall()
        return [{"type": "STOP", "attempt_id": row["attempt_id"], "job_id": row["job_id"]} for row in rows]

    def _ensure_job(self, job_id: str) -> None:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
                raise JobServiceError("JOB_NOT_FOUND", "job_id", f"Unknown job: {job_id}", 404)

    def _transition_tx(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        new_state: str,
        *,
        attempt_id: str | None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        row = connection.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobServiceError("JOB_NOT_FOUND", "job_id", f"Unknown job: {job_id}", 404)
        old_state = row["state"]
        if new_state not in STATE_TRANSITIONS.get(old_state, set()):
            raise JobServiceError(
                "INVALID_STATE_TRANSITION", "state", f"Transition {old_state} -> {new_state} is not allowed.", 409
            )
        now = self.clock.now()
        connection.execute("UPDATE jobs SET state = ?, updated_at = ? WHERE job_id = ?", (new_state, now, job_id))
        if attempt_id is not None:
            started_at = now if new_state == "RUNNING" else None
            ended_at = now if new_state in TERMINAL_STATES else None
            connection.execute(
                "UPDATE attempts SET state = ?, started_at = COALESCE(started_at, ?), "
                "ended_at = COALESCE(?, ended_at) WHERE attempt_id = ?",
                (new_state, started_at, ended_at, attempt_id),
            )
        self._event_tx(connection, job_id, attempt_id, "STATE_CHANGED", new_state, payload or {})

    def _fail_tx(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        attempt_id: str,
        category: str,
        code: str,
        message: str,
    ) -> None:
        self._transition_tx(
            connection,
            job_id,
            "FAILED",
            attempt_id=attempt_id,
            payload={"error": {"category": category, "code": code, "message": _redact(message)}},
        )
        connection.execute(
            "UPDATE jobs SET error_category = ?, error_code = ?, error_message = ? WHERE job_id = ?",
            (category, code, _redact(message), job_id),
        )
        connection.execute("DELETE FROM leases WHERE attempt_id = ?", (attempt_id,))

    def _event_tx(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        attempt_id: str | None,
        event_type: str,
        state: str | None,
        payload: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO events(job_id, attempt_id, event_type, state, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, attempt_id, event_type, state, _json(payload), self.clock.now()),
        )


class FakeWorkerAgent:
    """Deterministic Agent used for local/cloud contract and restart tests."""

    def __init__(self, service: TrainingJobService, node_id: str):
        self.service = service
        self.node_id = node_id

    def run_next(self, *, succeed: bool = True) -> dict[str, Any] | None:
        assignment = self.service.poll_assignment(self.node_id)
        if assignment is None:
            return None
        attempt_id = assignment["attempt_id"]
        self.service.report_state(
            self.node_id,
            attempt_id,
            "RUNNING",
            metrics={"loss": 1.0, "learning_rate": 0.0001, "epoch": 1.0, "progress": 0.5, "gpu_memory_bytes": None},
        )
        output_dir = Path(assignment["config"]["output_dir"])
        job = self.service.get_job(assignment["job_id"])
        artifact_metadata = {
            "fake": True,
            "base_model_id": job["model_id"],
            "dataset_snapshot_id": job["dataset_snapshot_id"],
            "template": assignment["config"].get("template"),
            "adapter_type": "qlora" if assignment["config"].get("quantization_bit") == 4 else "lora",
            "code_version": job["code_version"],
            "config_fingerprint": job["config_fingerprint"],
        }
        self.service.append_logs(
            self.node_id,
            attempt_id,
            [{"stream": "stdout", "message": "fake executor completed metadata-only SFT"}],
        )
        if succeed:
            checkpoint_path = output_dir / "checkpoint-fake-0001"
            adapter_path = output_dir / "adapter_model.fake.safetensors"
            log_path = output_dir / "training.fake.log"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / "checkpoint.fake.json").write_text(
                json.dumps({**artifact_metadata, "step": 1}, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            adapter_path.write_text(
                json.dumps(
                    {**artifact_metadata, "warning": "No model weights are present."},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            log_path.write_text("fake executor completed metadata-only SFT\n", encoding="utf-8")
            self.service.add_artifacts(
                self.node_id,
                attempt_id,
                [
                    {
                        "kind": "checkpoint",
                        "path": str(checkpoint_path),
                        "bytes": (checkpoint_path / "checkpoint.fake.json").stat().st_size,
                        "sha256": hashlib.sha256(
                            (checkpoint_path / "checkpoint.fake.json").read_bytes()
                        ).hexdigest(),
                        "metadata": {**artifact_metadata, "step": 1},
                        "restorable": True,
                    },
                    {
                        "kind": "adapter",
                        "path": str(adapter_path),
                        "bytes": adapter_path.stat().st_size,
                        "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                        "metadata": artifact_metadata,
                    },
                    {
                        "kind": "log",
                        "path": str(log_path),
                        "bytes": log_path.stat().st_size,
                        "metadata": {"fake": True},
                    },
                ],
            )
            return self.service.report_state(self.node_id, attempt_id, "SUCCEEDED")
        return self.service.report_state(
            self.node_id,
            attempt_id,
            "FAILED",
            error_category="UNKNOWN",
            error_code="FAKE_EXECUTOR_FAILURE",
            error_message="Deterministic fake failure.",
        )
