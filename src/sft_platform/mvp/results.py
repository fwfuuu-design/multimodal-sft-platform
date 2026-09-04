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

"""Stage-5 result, fake evaluation, chat, and Adapter merge framework.

No implementation in this module opens model weights. ``FakeModelAdapter`` is
an injectable contract test double and all generated artifacts explicitly carry
``fake: true`` metadata.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from .jobs import RandomIdGenerator, SystemClock, TrainingJobService


class ResultServiceError(ValueError):
    def __init__(self, code: str, field: str, message: str, status_code: int = 422):
        self.code = code
        self.field = field
        self.status_code = status_code
        super().__init__(f"{code}: {message} (field={field!r})")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "field": self.field, "message": str(self)}


class ModelAdapter(Protocol):
    def complete(
        self,
        model: Mapping[str, Any],
        artifact: Mapping[str, Any] | None,
        messages: Sequence[Mapping[str, Any]],
        *,
        template: str,
        tokenizer: str,
        image_processor: str | None,
    ) -> dict[str, Any]: ...

    def predict(
        self,
        model: Mapping[str, Any],
        artifact: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        *,
        template: str,
    ) -> tuple[dict[str, float | int], list[dict[str, Any]]]: ...


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _flatten_chat_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, str]], int]:
    if not messages:
        raise ResultServiceError("CHAT_MESSAGES_REQUIRED", "messages", "At least one message is required.")
    normalized = []
    image_count = 0
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ResultServiceError("CHAT_ROLE_INVALID", f"messages[{message_index}].role", "Chat role is invalid.")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts = []
            for item_index, item in enumerate(content):
                if not isinstance(item, Mapping):
                    raise ResultServiceError(
                        "CHAT_CONTENT_INVALID", f"messages[{message_index}].content[{item_index}]", "Content item is invalid."
                    )
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item_type == "image" and isinstance(item.get("asset_id"), str):
                    if item["asset_id"].startswith(("http://", "https://", "data:")):
                        raise ResultServiceError(
                            "CHAT_IMAGE_REFERENCE_INVALID", "asset_id", "Remote and data image references are forbidden."
                        )
                    parts.append("<image>")
                    image_count += 1
                else:
                    raise ResultServiceError(
                        "CHAT_MODALITY_NOT_SUPPORTED",
                        f"messages[{message_index}].content[{item_index}]",
                        "Only text and controlled image items are supported.",
                    )
            text = "".join(parts)
        else:
            raise ResultServiceError("CHAT_CONTENT_INVALID", f"messages[{message_index}].content", "Content is invalid.")
        normalized.append({"role": str(role), "content": text})
    if normalized[-1]["role"] != "user":
        raise ResultServiceError("CHAT_LAST_MESSAGE_INVALID", "messages", "The last message must be from the user.")
    return normalized, image_count


class FakeModelAdapter:
    """Deterministic model-loading boundary for tests and product demos."""

    def _validate_load(
        self,
        model: Mapping[str, Any],
        artifact: Mapping[str, Any] | None,
        *,
        template: str,
        tokenizer: str,
        image_processor: str | None,
        needs_images: bool,
    ) -> None:
        expected_template = model.get("template")
        if expected_template and template != expected_template:
            raise ResultServiceError("MODEL_TEMPLATE_MISMATCH", "template", "Template does not match model metadata.")
        if tokenizer != "auto":
            raise ResultServiceError("TOKENIZER_NOT_AVAILABLE", "tokenizer", "Fake adapter only accepts validated auto tokenizer.")
        modes = model.get("compatibility", {}).get("modes", [])
        if needs_images and "image_text_sft" not in modes:
            raise ResultServiceError("MODEL_IMAGE_INPUT_NOT_SUPPORTED", "messages", "Model does not support image input.")
        if needs_images and image_processor != "auto":
            raise ResultServiceError(
                "IMAGE_PROCESSOR_NOT_AVAILABLE", "image_processor", "Image model requires the validated auto processor."
            )
        if artifact is not None:
            metadata = artifact.get("metadata", {})
            if metadata.get("base_model_id") != model.get("id"):
                raise ResultServiceError("ADAPTER_BASE_MODEL_MISMATCH", "artifact_id", "Adapter belongs to another model.")
            if metadata.get("template") and metadata["template"] != template:
                raise ResultServiceError("ADAPTER_TEMPLATE_MISMATCH", "artifact_id", "Adapter template does not match.")

    def complete(
        self,
        model: Mapping[str, Any],
        artifact: Mapping[str, Any] | None,
        messages: Sequence[Mapping[str, Any]],
        *,
        template: str,
        tokenizer: str,
        image_processor: str | None,
    ) -> dict[str, Any]:
        normalized, image_count = _flatten_chat_messages(messages)
        self._validate_load(
            model,
            artifact,
            template=template,
            tokenizer=tokenizer,
            image_processor=image_processor,
            needs_images=image_count > 0,
        )
        digest = _fingerprint(
            {
                "model_id": model["id"],
                "artifact_id": artifact.get("id") if artifact else None,
                "messages": normalized,
            }
        )[:16]
        return {
            "model_id": model["id"],
            "artifact_id": artifact.get("id") if artifact else None,
            "modality": "image_text" if image_count else "text",
            "message": {"role": "assistant", "content": f"fake-response-{digest}"},
            "usage": {
                "prompt_characters": sum(len(message["content"]) for message in normalized),
                "completion_characters": len(f"fake-response-{digest}"),
                "image_count": image_count,
            },
            "fake": True,
        }

    def predict(
        self,
        model: Mapping[str, Any],
        artifact: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        *,
        template: str,
    ) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
        needs_images = any(record.get("images") for record in records)
        self._validate_load(
            model,
            artifact,
            template=template,
            tokenizer="auto",
            image_processor="auto" if needs_images else None,
            needs_images=needs_images,
        )
        predictions = []
        for index, record in enumerate(records):
            digest = _fingerprint(
                {"model_id": model["id"], "artifact_id": artifact["id"], "index": index, "record": record}
            )[:16]
            predictions.append({"sample_index": index, "prediction": f"fake-prediction-{digest}", "fake": True})
        metrics: dict[str, float | int] = {
            "sample_count": len(records),
            "eval_loss": 1.0,
            "exact_match": 0.5 if records else 0.0,
        }
        return metrics, predictions


class ResultService:
    def __init__(
        self,
        database_path: str | Path,
        state_root: str | Path,
        *,
        model_service: Any,
        dataset_service: Any,
        training_service: TrainingJobService,
        model_adapter: ModelAdapter | None = None,
        clock: Any | None = None,
        id_generator: Callable[[str], str] | None = None,
    ):
        self.database_path = str(Path(database_path).expanduser().resolve())
        self.state_root = Path(state_root).expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.model_service = model_service
        self.dataset_service = dataset_service
        self.training_service = training_service
        self.model_adapter = model_adapter or FakeModelAdapter()
        self.clock = clock or SystemClock()
        self.id_generator = id_generator or RandomIdGenerator()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    evaluation_id TEXT PRIMARY KEY, state TEXT NOT NULL, model_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL, dataset_snapshot_id TEXT NOT NULL, config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL, prediction_path TEXT, manifest_path TEXT,
                    error_code TEXT, error_message TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS export_runs (
                    export_id TEXT PRIMARY KEY, state TEXT NOT NULL, model_id TEXT NOT NULL,
                    adapter_id TEXT NOT NULL, node_id TEXT NOT NULL, config_json TEXT NOT NULL,
                    requirements_json TEXT NOT NULL, output_path TEXT, manifest_path TEXT,
                    error_code TEXT, error_message TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                """
            )

    def training_manifest(self, job_id: str) -> dict[str, Any]:
        try:
            job = self.training_service.get_job(job_id)
            artifacts = self.training_service.list_artifacts(job_id)
        except Exception as error:
            raise ResultServiceError("JOB_NOT_FOUND", "job_id", "Training job does not exist.", 404) from error
        events = self.training_service.get_events(job_id)["items"]
        training_curve = [
            {"sequence": event["sequence"], **event["payload"]["metrics"]}
            for event in events
            if isinstance(event.get("payload", {}).get("metrics"), dict)
        ]
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifacts:
            by_kind.setdefault(artifact["kind"], []).append(artifact)
        complete = job["state"] == "SUCCEEDED" and all(kind in by_kind for kind in ("adapter", "checkpoint", "log"))
        return {
            "job_id": job_id,
            "state": job["state"],
            "complete": complete,
            "model_id": job["model_id"],
            "dataset_snapshot_id": job["dataset_snapshot_id"],
            "code_version": job["code_version"],
            "config_snapshot": job["config_snapshot"],
            "config_fingerprint": job["config_fingerprint"],
            "attempts": job["attempts"],
            "logs_url": f"/api/v1/training-jobs/{job_id}/logs?download=true",
            "training_curve": training_curve,
            "artifacts": deepcopy(artifacts),
        }

    def downloadable_artifact(self, job_id: str, artifact_id: str) -> dict[str, Any]:
        artifact = self._artifact(artifact_id, allowed_kinds={"adapter"})
        if artifact["job_id"] != job_id:
            raise ResultServiceError("ARTIFACT_JOB_MISMATCH", "artifact_id", "Artifact belongs to another job.", 404)
        path = Path(artifact["path"])
        if not path.is_file():
            raise ResultServiceError(
                "ARTIFACT_FILE_NOT_FOUND", "artifact_id", "Artifact file is not available for download.", 404
            )
        return artifact

    def chat(
        self,
        *,
        model_id: str,
        artifact_id: str | None,
        messages: Sequence[Mapping[str, Any]],
        template: str | None = None,
        tokenizer: str = "auto",
        image_processor: str | None = "auto",
    ) -> dict[str, Any]:
        model = self._model(model_id)
        artifact = self._artifact(artifact_id, allowed_kinds={"adapter", "checkpoint"}) if artifact_id else None
        return self.model_adapter.complete(
            model,
            artifact,
            messages,
            template=template or model.get("template") or "default",
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

    def evaluate(
        self,
        *,
        model_id: str,
        artifact_id: str,
        dataset_snapshot_id: str,
        template: str | None = None,
        predict: bool = True,
    ) -> dict[str, Any]:
        model = self._model(model_id)
        artifact = self._artifact(artifact_id, allowed_kinds={"adapter", "checkpoint"})
        try:
            records = self.dataset_service.get_snapshot(dataset_snapshot_id)
        except Exception as error:
            raise ResultServiceError(
                "DATASET_SNAPSHOT_NOT_FOUND", "dataset_snapshot_id", "Evaluation dataset snapshot does not exist.", 404
            ) from error
        evaluation_id = self.id_generator("evaluation")
        now = self.clock.now()
        config = {
            "model_id": model_id,
            "artifact_id": artifact_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "template": template or model.get("template") or "default",
            "predict": predict,
            "adapter": "fake",
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_runs VALUES (?, 'RUNNING', ?, ?, ?, ?, '{}', NULL, NULL, NULL, NULL, ?, ?)",
                (evaluation_id, model_id, artifact_id, dataset_snapshot_id, _json(config), now, now),
            )
        try:
            metrics, predictions = self.model_adapter.predict(
                model, artifact, records, template=config["template"]
            )
            run_root = self.state_root / "evaluations" / evaluation_id
            prediction_path = run_root / "predictions.fake.jsonl"
            manifest_path = run_root / "manifest.json"
            if predict:
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_path.write_text(
                    "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in predictions),
                    encoding="utf-8",
                )
            manifest = {
                "id": evaluation_id,
                "state": "SUCCEEDED",
                "config": config,
                "metrics": metrics,
                "prediction_path": str(prediction_path) if predict else None,
                "fake": True,
            }
            _write_json_atomic(manifest_path, manifest)
            with self._connect() as connection:
                connection.execute(
                    "UPDATE evaluation_runs SET state = 'SUCCEEDED', metrics_json = ?, prediction_path = ?, "
                    "manifest_path = ?, updated_at = ? WHERE evaluation_id = ?",
                    (_json(metrics), str(prediction_path) if predict else None, str(manifest_path), self.clock.now(), evaluation_id),
                )
        except ResultServiceError as error:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE evaluation_runs SET state = 'FAILED', error_code = ?, error_message = ?, updated_at = ? "
                    "WHERE evaluation_id = ?",
                    (error.code, str(error), self.clock.now(), evaluation_id),
                )
            raise
        return self.get_evaluation(evaluation_id)

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE evaluation_id = ?", (evaluation_id,)
            ).fetchone()
        if row is None:
            raise ResultServiceError("EVALUATION_NOT_FOUND", "evaluation_id", "Evaluation run does not exist.", 404)
        return {
            "id": row["evaluation_id"],
            "state": row["state"],
            "model_id": row["model_id"],
            "artifact_id": row["artifact_id"],
            "dataset_snapshot_id": row["dataset_snapshot_id"],
            "config": json.loads(row["config_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "prediction_path": row["prediction_path"],
            "manifest_path": row["manifest_path"],
            "error": ({"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "fake": True,
        }

    def export_adapter(
        self,
        *,
        model_id: str,
        adapter_id: str,
        node_id: str,
        output_name: str,
    ) -> dict[str, Any]:
        if not output_name or "/" in output_name or "\\" in output_name or output_name in {".", ".."}:
            raise ResultServiceError("EXPORT_NAME_INVALID", "output_name", "Export name must be one safe path segment.")
        model = self._model(model_id)
        adapter = self._artifact(adapter_id, allowed_kinds={"adapter"})
        metadata = adapter.get("metadata", {})
        if metadata.get("base_model_id") != model_id:
            raise ResultServiceError("ADAPTER_BASE_MODEL_MISMATCH", "adapter_id", "Adapter belongs to another model.")
        if metadata.get("adapter_type") not in {"lora", "qlora"}:
            raise ResultServiceError("ADAPTER_TYPE_NOT_SUPPORTED", "adapter_id", "Only LoRA and QLoRA adapters can merge.")
        node = self.training_service.get_node(node_id)
        requirements = self._export_requirements(model, metadata, node)
        export_id = self.id_generator("export")
        now = self.clock.now()
        output_root = (Path(node["work_roots"]["outputs"]) / "exports" / export_id).resolve()
        if not output_root.is_relative_to(Path(node["work_roots"]["outputs"])):
            raise ResultServiceError("EXPORT_PATH_INVALID", "output_name", "Export path escaped the node output root.")
        output_path = output_root / output_name
        manifest_path = output_root / "merge-manifest.json"
        config = {
            "operation": "merge_adapter",
            "model_id": model_id,
            "adapter_id": adapter_id,
            "node_id": node_id,
            "output_name": output_name,
            "adapter_type": metadata["adapter_type"],
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO export_runs VALUES (?, 'RUNNING', ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)",
                (export_id, model_id, adapter_id, node_id, _json(config), _json(requirements), now, now),
            )
        output_path.mkdir(parents=True, exist_ok=True)
        fake_marker = output_path / "merged-model.fake.json"
        _write_json_atomic(
            fake_marker,
            {
                "fake": True,
                "model_id": model_id,
                "adapter_id": adapter_id,
                "warning": "No model weights were read or merged.",
            },
        )
        manifest = {
            "id": export_id,
            "state": "SUCCEEDED",
            "config": config,
            "requirements": requirements,
            "artifacts": [{"kind": "merged_model_manifest", "path": str(fake_marker), "fake": True}],
            "fake": True,
        }
        _write_json_atomic(manifest_path, manifest)
        with self._connect() as connection:
            connection.execute(
                "UPDATE export_runs SET state = 'SUCCEEDED', output_path = ?, manifest_path = ?, updated_at = ? "
                "WHERE export_id = ?",
                (str(output_path), str(manifest_path), self.clock.now(), export_id),
            )
        return self.get_export(export_id)

    def get_export(self, export_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM export_runs WHERE export_id = ?", (export_id,)).fetchone()
        if row is None:
            raise ResultServiceError("EXPORT_NOT_FOUND", "export_id", "Export run does not exist.", 404)
        return {
            "id": row["export_id"],
            "state": row["state"],
            "model_id": row["model_id"],
            "adapter_id": row["adapter_id"],
            "node_id": row["node_id"],
            "config": json.loads(row["config_json"]),
            "requirements": json.loads(row["requirements_json"]),
            "output_path": row["output_path"],
            "manifest_path": row["manifest_path"],
            "error": ({"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "fake": True,
        }

    def _model(self, model_id: str) -> dict[str, Any]:
        try:
            model = self.model_service.get(model_id)
        except Exception as error:
            raise ResultServiceError("MODEL_NOT_FOUND", "model_id", "Registered model does not exist.", 404) from error
        if not model.get("compatibility", {}).get("mvp_compatible"):
            raise ResultServiceError("MODEL_NOT_MVP_COMPATIBLE", "model_id", "Model is outside MVP text/image SFT.")
        return model

    def _artifact(self, artifact_id: str | None, *, allowed_kinds: set[str]) -> dict[str, Any]:
        if not artifact_id:
            raise ResultServiceError("ARTIFACT_REQUIRED", "artifact_id", "Artifact ID is required.")
        try:
            artifact = self.training_service.get_artifact(artifact_id)
        except Exception as error:
            raise ResultServiceError("ARTIFACT_NOT_FOUND", "artifact_id", "Artifact does not exist.", 404) from error
        if artifact["kind"] not in allowed_kinds:
            raise ResultServiceError(
                "ARTIFACT_KIND_NOT_SUPPORTED", "artifact_id", f"Expected one of: {sorted(allowed_kinds)}."
            )
        return artifact

    def _export_requirements(
        self, model: Mapping[str, Any], adapter_metadata: Mapping[str, Any], node: Mapping[str, Any]
    ) -> dict[str, Any]:
        model_bytes = model.get("estimated_model_bytes")
        if model_bytes is None and model.get("parameter_count"):
            bytes_per_parameter = 4 if model.get("weight_precision") == "fp32" else 2
            model_bytes = model["parameter_count"] * bytes_per_parameter
        if not model_bytes:
            raise ResultServiceError(
                "MODEL_SIZE_UNKNOWN", "model_id", "Model size metadata is required before Adapter merge.", 409
            )
        if adapter_metadata.get("adapter_type") == "qlora" and model.get("weight_precision") not in {
            "fp16",
            "bf16",
            "fp32",
        }:
            raise ResultServiceError(
                "BASE_MODEL_PRECISION_UNSUPPORTED",
                "model_id",
                "QLoRA merge requires an unquantized fp16, bf16, or fp32 base model.",
                409,
            )
        requirements = {
            "model_bytes": model_bytes,
            "required_system_memory_bytes": model_bytes * 2,
            "required_gpu_memory_bytes": model_bytes,
            "required_disk_bytes": model_bytes * 2,
        }
        if node.get("system_memory_free_bytes") is None:
            raise ResultServiceError(
                "RESOURCE_REPORT_INCOMPLETE", "node_id", "Node must report free system memory before export.", 409
            )
        if node["system_memory_free_bytes"] < requirements["required_system_memory_bytes"]:
            raise ResultServiceError("SYSTEM_MEMORY_INSUFFICIENT", "node_id", "Node has insufficient system memory.", 409)
        if node.get("disk_free_bytes") is None or node["disk_free_bytes"] < requirements["required_disk_bytes"]:
            raise ResultServiceError("DISK_INSUFFICIENT", "node_id", "Node has insufficient disk space.", 409)
        available_gpu_memory = sum(gpu["memory_free_bytes"] for gpu in node.get("gpus", []) if gpu["lease"] is None)
        if available_gpu_memory < requirements["required_gpu_memory_bytes"]:
            raise ResultServiceError("GPU_MEMORY_INSUFFICIENT", "node_id", "Node has insufficient free GPU memory.", 409)
        return requirements
