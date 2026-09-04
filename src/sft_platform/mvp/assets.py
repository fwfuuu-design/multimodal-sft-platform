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

"""Model and dataset contracts for the text/image SFT MVP.

The services in this module inspect metadata and build immutable manifests.
They never download model weights, initialize a model, or start training.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, BinaryIO, Protocol


MAX_UPLOAD_FILE_BYTES = 5 * 1024**3
MAX_DATASET_BYTES = 100 * 1024**3
MAX_IMAGE_BYTES = 20 * 1024**2
MAX_IMAGE_DIMENSION = 4096
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_MODEL_SOURCES = {"local", "huggingface", "modelscope"}
SUPPORTED_DATA_FORMATS = {"alpaca", "sharegpt", "openai"}
_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class AssetValidationError(ValueError):
    def __init__(self, code: str, field: str, message: str, sample_index: int | None = None):
        self.code = code
        self.field = field
        self.sample_index = sample_index
        location = f", sample_index={sample_index}" if sample_index is not None else ""
        super().__init__(f"{code}: {message} (field={field!r}{location})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "sample_index": self.sample_index,
            "message": str(self),
        }


@dataclass(frozen=True)
class ModelReference:
    name: str
    source: str
    location: str
    credential_id: str | None = None


class HubInspector(Protocol):
    def inspect(self, source: str, location: str, credential_id: str | None) -> Mapping[str, Any]: ...


class FakeHub:
    """Deterministic Hugging Face/ModelScope metadata stub."""

    def __init__(self, records: Mapping[tuple[str, str], Mapping[str, Any]] | None = None):
        self._records = {key: deepcopy(dict(value)) for key, value in (records or {}).items()}
        self.requests: list[dict[str, Any]] = []

    def add(self, source: str, location: str, metadata: Mapping[str, Any]) -> None:
        self._records[(source, location)] = deepcopy(dict(metadata))

    def inspect(self, source: str, location: str, credential_id: str | None) -> Mapping[str, Any]:
        self.requests.append(
            {
                "source": source,
                "location": location,
                "credential_id": credential_id,
            }
        )
        try:
            return deepcopy(self._records[(source, location)])
        except KeyError as err:
            raise AssetValidationError(
                "MODEL_NOT_FOUND",
                "location",
                f"Fake Hub has no metadata for {source}:{location}",
            ) from err


class ModelAssetService:
    def __init__(
        self,
        model_roots: Sequence[str | Path],
        hub_inspector: HubInspector | None = None,
        state_root: str | Path | None = None,
    ):
        if not model_roots:
            raise ValueError("At least one controlled model root is required.")
        self._model_roots = tuple(Path(root).expanduser().resolve() for root in model_roots)
        self._hub_inspector = hub_inspector or FakeHub()
        self._state_file = Path(state_root).expanduser().resolve() / "models.json" if state_root else None
        self._models: dict[str, dict[str, Any]] = _load_json_mapping(self._state_file)

    def register(self, reference: ModelReference) -> dict[str, Any]:
        metadata = self.validate(reference)
        fingerprint = _fingerprint(
            {
                "name": reference.name,
                "source": reference.source,
                "location": metadata["location"],
                "revision": metadata.get("revision"),
            }
        )
        model_id = f"model-{fingerprint[:16]}"
        record = {"id": model_id, "name": reference.name, **metadata}
        self._models[model_id] = record
        _write_json_atomic(self._state_file, self._models)
        return deepcopy(record)

    def validate(self, reference: ModelReference) -> dict[str, Any]:
        source = reference.source.lower()
        if source not in SUPPORTED_MODEL_SOURCES:
            raise AssetValidationError(
                "MODEL_SOURCE_NOT_SUPPORTED",
                "source",
                "Model source must be local, huggingface, or modelscope.",
            )
        if not reference.name.strip():
            raise AssetValidationError("MODEL_NAME_REQUIRED", "name", "Model name is required.")
        if _looks_like_secret(reference.credential_id):
            raise AssetValidationError(
                "PLAINTEXT_CREDENTIAL_NOT_ALLOWED",
                "credential_id",
                "Use an opaque credential ID instead of a token.",
            )

        if source == "local":
            metadata = self._inspect_local(reference.location)
            location = str(Path(reference.location).expanduser().resolve())
            cache_state = "available"
        else:
            if not _REPOSITORY_ID.fullmatch(reference.location):
                raise AssetValidationError(
                    "MODEL_REPOSITORY_INVALID",
                    "location",
                    "Hub model location must be a repository ID, not a URL or path.",
                )
            metadata = dict(self._hub_inspector.inspect(source, reference.location, reference.credential_id))
            if metadata.get("private") is True and reference.credential_id is None:
                raise AssetValidationError(
                    "MODEL_CREDENTIAL_REQUIRED",
                    "credential_id",
                    "Private repositories require an opaque credential ID.",
                )
            location = reference.location
            cache_state = str(metadata.get("cache_state", "not_cached"))

        modalities = _normalize_modalities(metadata.get("modalities", ["text"]))
        compatibility = _model_compatibility(modalities)
        return {
            "source": source,
            "location": location,
            "revision": metadata.get("revision", "main"),
            "cache_state": cache_state,
            "parameter_count": _optional_positive_int(metadata.get("parameter_count"), "parameter_count"),
            "estimated_model_bytes": _optional_positive_int(
                metadata.get("estimated_model_bytes"), "estimated_model_bytes"
            ),
            "weight_precision": _normalize_weight_precision(metadata.get("weight_precision")),
            "template": metadata.get("template"),
            "context_length": _optional_positive_int(metadata.get("context_length"), "context_length"),
            "modalities": modalities,
            "compatibility": compatibility,
        }

    def list(self) -> list[dict[str, Any]]:
        return [deepcopy(self._models[key]) for key in sorted(self._models)]

    def get(self, model_id: str) -> dict[str, Any]:
        try:
            return deepcopy(self._models[model_id])
        except KeyError as err:
            raise AssetValidationError("MODEL_NOT_FOUND", "model_id", f"Unknown model ID: {model_id}") from err

    def _inspect_local(self, location: str) -> dict[str, Any]:
        model_path = Path(location).expanduser().resolve()
        if not any(model_path.is_relative_to(root) for root in self._model_roots):
            raise AssetValidationError(
                "MODEL_PATH_OUTSIDE_ROOT",
                "location",
                "Local model path must be inside a controlled model root.",
            )
        if not model_path.is_dir():
            raise AssetValidationError("MODEL_PATH_NOT_FOUND", "location", "Local model directory does not exist.")

        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise AssetValidationError("MODEL_CONFIG_NOT_FOUND", "location", "Local model config.json is required.")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            raise AssetValidationError("MODEL_CONFIG_INVALID", "location", "Local config.json is invalid.") from err
        if not isinstance(config, dict):
            raise AssetValidationError("MODEL_CONFIG_INVALID", "location", "Local config.json must be an object.")

        return {
            "revision": config.get("_commit_hash", "local"),
            "parameter_count": config.get("parameter_count"),
            "estimated_model_bytes": config.get("estimated_model_bytes"),
            "weight_precision": config.get("weight_precision") or config.get("torch_dtype"),
            "template": config.get("chat_template_name") or config.get("template"),
            "context_length": config.get("max_position_embeddings"),
            "modalities": config.get("modalities", ["text"]),
        }


def build_upstream_model_catalog() -> list[dict[str, Any]]:
    """Expose every upstream registration with a conservative MVP marker."""
    from ..extras.constants import DEFAULT_TEMPLATE, MULTIMODAL_SUPPORTED_MODELS, SUPPORTED_MODELS

    review_tokens = ("audio", "omni", "video")
    catalog = []
    for name, source_paths in SUPPORTED_MODELS.items():
        multimodal = name in MULTIMODAL_SUPPORTED_MODELS
        needs_review = multimodal and any(token in name.lower() for token in review_tokens)
        catalog.append(
            {
                "name": name,
                "sources": {str(source): path for source, path in source_paths.items()},
                "template": DEFAULT_TEMPLATE[name] or None,
                "modality": "image_text" if multimodal else "text",
                "mvp_compatible": not needs_review,
                "compatibility_reason": (
                    "audio/video/omni registration requires an explicit image-text compatibility override"
                    if needs_review
                    else None
                ),
            }
        )
    return catalog


@dataclass
class DatasetInspection:
    source_path: str | None
    source_paths: list[str]
    formatting: str
    ready: bool
    snapshot_id: str | None
    statistics: dict[str, Any]
    preview: list[dict[str, Any]]
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAssetService:
    def __init__(
        self,
        dataset_roots: Sequence[str | Path],
        media_roots: Sequence[str | Path],
        state_root: str | Path | None = None,
    ):
        if not dataset_roots or not media_roots:
            raise ValueError("Controlled dataset and media roots are required.")
        self._dataset_roots = tuple(Path(root).expanduser().resolve() for root in dataset_roots)
        self._media_roots = tuple(Path(root).expanduser().resolve() for root in media_roots)
        self._state_root = Path(state_root).expanduser().resolve() if state_root else None
        uploaded = _load_json_mapping(self._state_root / "dataset-files.json" if self._state_root else None)
        self._snapshots: dict[str, list[dict[str, Any]]] = {}
        self._uploaded_files = {
            file_id: Path(path).resolve()
            for file_id, path in uploaded.items()
            if isinstance(path, str)
            and Path(path).resolve().is_file()
            and any(Path(path).resolve().is_relative_to(root) for root in self._dataset_roots)
        }

    def store_upload(self, filename: str, stream: BinaryIO) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise AssetValidationError(
                "DATASET_FILE_TYPE_NOT_SUPPORTED",
                "filename",
                "Only JSON and JSONL files are supported in the MVP.",
            )
        upload_root = self._dataset_roots[0]
        upload_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".upload-", dir=upload_root, delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_FILE_BYTES:
                        raise AssetValidationError(
                            "DATASET_FILE_TOO_LARGE",
                            "file",
                            f"Dataset file exceeds {MAX_UPLOAD_FILE_BYTES} bytes.",
                        )
                    digest.update(chunk)
                    temporary_file.write(chunk)
            sha256 = digest.hexdigest()
            file_id = f"dataset-file-{sha256[:20]}"
            destination = upload_root / f"{file_id}{suffix}"
            if destination.exists():
                temporary_path.unlink()
            else:
                temporary_path.replace(destination)
            self._uploaded_files[file_id] = destination
            _write_json_atomic(
                self._state_root / "dataset-files.json" if self._state_root else None,
                {key: str(value) for key, value in self._uploaded_files.items()},
            )
            return {"id": file_id, "filename": Path(filename).name, "bytes": size, "sha256": sha256}
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise

    def inspect_upload(
        self,
        file_id: str,
        formatting: str,
        mapping: Mapping[str, str] | None = None,
        preview_limit: int = 20,
    ) -> DatasetInspection:
        return self.inspect_uploads([file_id], formatting, mapping, preview_limit)

    def inspect_uploads(
        self,
        file_ids: Sequence[str],
        formatting: str,
        mapping: Mapping[str, str] | None = None,
        preview_limit: int = 20,
    ) -> DatasetInspection:
        if not file_ids or len(set(file_ids)) != len(file_ids):
            raise AssetValidationError(
                "DATASET_FILES_INVALID",
                "dataset_file_ids",
                "Dataset file IDs must be non-empty and unique.",
            )
        try:
            source_paths = [self._uploaded_files[file_id] for file_id in file_ids]
        except KeyError as err:
            raise AssetValidationError(
                "DATASET_FILE_NOT_FOUND",
                "dataset_file_ids",
                f"Unknown uploaded dataset file: {err.args[0]}",
            ) from err
        return self._inspect_paths(source_paths, formatting, mapping, preview_limit)

    def inspect(
        self,
        source_path: str | Path,
        formatting: str,
        mapping: Mapping[str, str] | None = None,
        preview_limit: int = 20,
    ) -> DatasetInspection:
        return self._inspect_paths([self._controlled_file(source_path)], formatting, mapping, preview_limit)

    def _inspect_paths(
        self,
        data_paths: Sequence[Path],
        formatting: str,
        mapping: Mapping[str, str] | None,
        preview_limit: int,
    ) -> DatasetInspection:
        normalized_format = formatting.lower()
        if normalized_format not in SUPPORTED_DATA_FORMATS:
            raise AssetValidationError(
                "DATASET_FORMAT_NOT_SUPPORTED",
                "formatting",
                "Dataset formatting must be alpaca, sharegpt, or openai.",
            )
        if preview_limit < 1 or preview_limit > 100:
            raise AssetValidationError("PREVIEW_LIMIT_INVALID", "preview_limit", "Preview limit must be 1..100.")

        file_sizes = [data_path.stat().st_size for data_path in data_paths]
        if any(file_size > MAX_UPLOAD_FILE_BYTES for file_size in file_sizes):
            raise AssetValidationError(
                "DATASET_FILE_TOO_LARGE",
                "source_paths",
                f"Each dataset file must not exceed {MAX_UPLOAD_FILE_BYTES} bytes.",
            )
        total_bytes = sum(file_sizes)
        if total_bytes > MAX_DATASET_BYTES:
            raise AssetValidationError(
                "DATASET_TOO_LARGE",
                "source_paths",
                f"Dataset exceeds {MAX_DATASET_BYTES} bytes in total.",
            )

        records = []
        for data_path in data_paths:
            records.extend(_read_records(data_path))
        canonical_records: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        image_metadata: list[dict[str, Any]] = []
        text_lengths: list[int] = []

        for sample_index, record in enumerate(records):
            try:
                canonical = self._convert_record(record, normalized_format, mapping or {}, sample_index)
                sample_images = []
                for image_index, image in enumerate(canonical["images"]):
                    image_info = self._inspect_image(image, sample_index, image_index)
                    sample_images.append(image_info["path"])
                    image_metadata.append(image_info)
                canonical["images"] = sample_images
                canonical_records.append(canonical)
                text_lengths.append(sum(len(message["content"]) for message in canonical["messages"]))
            except AssetValidationError as err:
                errors.append(err.to_dict())

        ready = len(errors) == 0 and len(canonical_records) > 0
        snapshot_id = None
        if ready:
            snapshot_id = f"dataset-{_fingerprint({'records': canonical_records, 'images': image_metadata})[:20]}"
            self._snapshots[snapshot_id] = deepcopy(canonical_records)
            _write_json_atomic(self._snapshot_path(snapshot_id), canonical_records)

        widths = [image["width"] for image in image_metadata]
        heights = [image["height"] for image in image_metadata]
        source_paths = [str(data_path) for data_path in data_paths]
        statistics = {
            "source_file_count": len(data_paths),
            "sample_count": len(records),
            "valid_sample_count": len(canonical_records),
            "anomaly_count": len(errors),
            "text_length": _number_stats(text_lengths),
            "image_count": len(image_metadata),
            "image_width": _number_stats(widths),
            "image_height": _number_stats(heights),
            "source_bytes": total_bytes,
        }
        return DatasetInspection(
            source_path=source_paths[0] if len(source_paths) == 1 else None,
            source_paths=source_paths,
            formatting=normalized_format,
            ready=ready,
            snapshot_id=snapshot_id,
            statistics=statistics,
            preview=deepcopy(canonical_records[:preview_limit]),
            errors=errors,
        )

    def get_snapshot(self, snapshot_id: str) -> list[dict[str, Any]]:
        try:
            return deepcopy(self._snapshots[snapshot_id])
        except KeyError:
            snapshot_path = self._snapshot_path(snapshot_id)
            if snapshot_path is not None and snapshot_path.is_file():
                try:
                    records = json.loads(snapshot_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as err:
                    raise AssetValidationError(
                        "DATASET_SNAPSHOT_INVALID", "snapshot_id", "Persisted dataset snapshot is invalid."
                    ) from err
                if isinstance(records, list) and all(isinstance(record, dict) for record in records):
                    self._snapshots[snapshot_id] = records
                    return deepcopy(records)
            raise AssetValidationError(
                "DATASET_SNAPSHOT_NOT_FOUND",
                "snapshot_id",
                f"Unknown dataset snapshot: {snapshot_id}",
            )

    def _snapshot_path(self, snapshot_id: str) -> Path | None:
        if self._state_root is None:
            return None
        if not re.fullmatch(r"dataset-[a-f0-9]{20}", snapshot_id):
            raise AssetValidationError("DATASET_SNAPSHOT_ID_INVALID", "snapshot_id", "Dataset snapshot ID is invalid.")
        return self._state_root / "dataset-snapshots" / f"{snapshot_id}.json"

    def _controlled_file(self, source_path: str | Path) -> Path:
        path = Path(source_path).expanduser().resolve()
        if not any(path.is_relative_to(root) for root in self._dataset_roots):
            raise AssetValidationError(
                "DATASET_PATH_OUTSIDE_ROOT",
                "source_path",
                "Dataset path must be inside a controlled dataset root.",
            )
        if not path.is_file():
            raise AssetValidationError("DATASET_FILE_NOT_FOUND", "source_path", "Dataset file does not exist.")
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise AssetValidationError(
                "DATASET_FILE_TYPE_NOT_SUPPORTED",
                "source_path",
                "Only JSON and JSONL files are supported in the MVP.",
            )
        return path

    def _convert_record(
        self,
        record: Any,
        formatting: str,
        mapping: Mapping[str, str],
        sample_index: int,
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise AssetValidationError(
                "DATASET_SAMPLE_INVALID", "sample", "Each dataset sample must be an object.", sample_index
            )
        if _mapped(record, mapping, "videos") or _mapped(record, mapping, "audios"):
            raise AssetValidationError(
                "DATASET_MODALITY_NOT_SUPPORTED",
                "videos/audios",
                "Audio and video fields are outside the MVP.",
                sample_index,
            )

        if formatting == "alpaca":
            messages, embedded_images = _convert_alpaca(record, mapping, sample_index)
        else:
            messages, embedded_images = _convert_messages(record, mapping, formatting, sample_index)

        listed_images = _normalize_image_list(_mapped(record, mapping, "images"), sample_index)
        images = embedded_images + listed_images
        placeholder_count = sum(message["content"].count("<image>") for message in messages)
        if placeholder_count != len(images):
            raise AssetValidationError(
                "IMAGE_PLACEHOLDER_MISMATCH",
                "images",
                f"Found {placeholder_count} <image> placeholders but {len(images)} image paths.",
                sample_index,
            )
        return {"messages": messages, "images": images}

    def _inspect_image(self, image: str, sample_index: int, image_index: int) -> dict[str, Any]:
        candidate_paths = []
        raw_path = Path(image).expanduser()
        if raw_path.is_absolute():
            candidate_paths.append(raw_path.resolve())
        else:
            candidate_paths.extend((root / raw_path).resolve() for root in self._media_roots)
        image_path = next(
            (
                path
                for path in candidate_paths
                if path.is_file() and any(path.is_relative_to(root) for root in self._media_roots)
            ),
            None,
        )
        field_name = f"images[{image_index}]"
        if image_path is None:
            raise AssetValidationError(
                "IMAGE_NOT_FOUND", field_name, f"Image does not exist in a controlled media root: {image}", sample_index
            )
        image_size = image_path.stat().st_size
        if image_size > MAX_IMAGE_BYTES:
            raise AssetValidationError(
                "IMAGE_TOO_LARGE",
                field_name,
                f"Image exceeds {MAX_IMAGE_BYTES} bytes.",
                sample_index,
            )

        try:
            from PIL import Image

            with Image.open(image_path) as image_file:
                image_format = str(image_file.format).upper()
                width, height = image_file.size
                image_file.verify()
        except Exception as err:
            raise AssetValidationError(
                "IMAGE_INVALID", field_name, f"Image cannot be decoded: {image}", sample_index
            ) from err
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise AssetValidationError(
                "IMAGE_FORMAT_NOT_SUPPORTED",
                field_name,
                "Only JPEG, PNG, and WebP images are supported.",
                sample_index,
            )
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise AssetValidationError(
                "IMAGE_DIMENSION_TOO_LARGE",
                field_name,
                f"Image dimensions must not exceed {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}.",
                sample_index,
            )
        return {
            "path": str(image_path),
            "format": image_format,
            "width": width,
            "height": height,
            "bytes": image_size,
            "sha256": _file_sha256(image_path),
        }


def _read_records(path: Path) -> list[Any]:
    try:
        if path.suffix.lower() == ".jsonl":
            records = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise AssetValidationError(
                        "DATASET_JSON_INVALID",
                        "source_path",
                        f"Invalid JSONL at line {line_number}.",
                    ) from err
            return records
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as err:
        raise AssetValidationError("DATASET_ENCODING_INVALID", "source_path", "Dataset must be UTF-8.") from err
    except json.JSONDecodeError as err:
        raise AssetValidationError("DATASET_JSON_INVALID", "source_path", "Dataset JSON is invalid.") from err
    if not isinstance(payload, list):
        raise AssetValidationError("DATASET_ROOT_INVALID", "source_path", "Dataset JSON root must be a list.")
    return payload


def _convert_alpaca(
    record: Mapping[str, Any], mapping: Mapping[str, str], sample_index: int
) -> tuple[list[dict[str, str]], list[str]]:
    instruction = _mapped(record, mapping, "instruction", "instruction")
    query = _mapped(record, mapping, "input", "input")
    response = _mapped(record, mapping, "output", "output")
    if not isinstance(instruction, str) or not isinstance(query or "", str) or not isinstance(response, str):
        raise AssetValidationError(
            "ALPACA_FIELDS_INVALID",
            "instruction/input/output",
            "Alpaca instruction and output must be strings; input must be a string when present.",
            sample_index,
        )
    messages = []
    history = _mapped(record, mapping, "history") or []
    if not isinstance(history, list):
        raise AssetValidationError("ALPACA_HISTORY_INVALID", "history", "Alpaca history must be a list.", sample_index)
    for turn in history:
        if not isinstance(turn, list) or len(turn) != 2 or not all(isinstance(item, str) for item in turn):
            raise AssetValidationError(
                "ALPACA_HISTORY_INVALID", "history", "Each history turn must contain two strings.", sample_index
            )
        messages.extend(({"role": "user", "content": turn[0]}, {"role": "assistant", "content": turn[1]}))
    prompt = "\n".join(part for part in (instruction, query) if part)
    messages.extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": response}))
    return messages, []


def _convert_messages(
    record: Mapping[str, Any],
    mapping: Mapping[str, str],
    formatting: str,
    sample_index: int,
) -> tuple[list[dict[str, str]], list[str]]:
    raw_messages = _mapped(record, mapping, "messages", "conversations" if formatting == "sharegpt" else "messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise AssetValidationError("MESSAGES_INVALID", "messages", "Messages must be a non-empty list.", sample_index)

    role_field = mapping.get("role", "from" if formatting == "sharegpt" else "role")
    content_field = mapping.get("content", "value" if formatting == "sharegpt" else "content")
    role_mapping = {
        mapping.get("user_role", "human" if formatting == "sharegpt" else "user"): "user",
        mapping.get("assistant_role", "gpt" if formatting == "sharegpt" else "assistant"): "assistant",
        mapping.get("system_role", "system"): "system",
    }
    messages: list[dict[str, str]] = []
    images: list[str] = []
    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, dict) or message.get(role_field) not in role_mapping:
            raise AssetValidationError(
                "MESSAGE_ROLE_INVALID",
                f"messages[{message_index}].{role_field}",
                "Only system, user, and assistant roles are supported.",
                sample_index,
            )
        content, content_images = _flatten_content(message.get(content_field), sample_index, message_index)
        messages.append({"role": role_mapping[message[role_field]], "content": content})
        images.extend(content_images)

    conversational = messages[1:] if messages[0]["role"] == "system" else messages
    if not conversational or len(conversational) % 2 != 0:
        raise AssetValidationError(
            "MESSAGE_COUNT_INVALID", "messages", "Messages must contain complete user/assistant pairs.", sample_index
        )
    for index, message in enumerate(conversational):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise AssetValidationError(
                "MESSAGE_ORDER_INVALID",
                f"messages[{index}]",
                "Messages must alternate between user and assistant.",
                sample_index,
            )
    return messages, images


def _flatten_content(content: Any, sample_index: int, message_index: int) -> tuple[str, list[str]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise AssetValidationError(
            "MESSAGE_CONTENT_INVALID",
            f"messages[{message_index}].content",
            "Message content must be text or an OpenAI multimodal content list.",
            sample_index,
        )
    parts: list[str] = []
    images: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            raise AssetValidationError(
                "MESSAGE_CONTENT_INVALID",
                f"messages[{message_index}].content",
                "Content items must be objects.",
                sample_index,
            )
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", item.get("value"))
            if not isinstance(text, str):
                raise AssetValidationError(
                    "MESSAGE_CONTENT_INVALID", "text", "Text content value must be a string.", sample_index
                )
            parts.append(text)
        elif item_type in {"image", "image_url"}:
            image_value = item.get("value", item.get("image_url"))
            if isinstance(image_value, dict):
                image_value = image_value.get("url")
            if not isinstance(image_value, str) or image_value.startswith(("http://", "https://", "data:")):
                raise AssetValidationError(
                    "IMAGE_REFERENCE_INVALID",
                    "image_url",
                    "Images must use controlled local paths; remote/data URLs are not accepted.",
                    sample_index,
                )
            parts.append("<image>")
            images.append(image_value)
        else:
            raise AssetValidationError(
                "MESSAGE_CONTENT_TYPE_NOT_SUPPORTED",
                "content.type",
                "Only text and image_url content items are supported.",
                sample_index,
            )
    return "".join(parts), images


def _mapped(
    record: Mapping[str, Any],
    mapping: Mapping[str, str],
    canonical_name: str,
    default_source_name: str | None = None,
) -> Any:
    source_name = mapping.get(canonical_name, default_source_name or canonical_name)
    return record.get(source_name)


def _normalize_image_list(value: Any, sample_index: int) -> list[str]:
    if value in (None, "", []):
        return []
    images = [value] if isinstance(value, str) else value
    if not isinstance(images, list) or not all(isinstance(image, str) and image for image in images):
        raise AssetValidationError("IMAGE_FIELD_INVALID", "images", "Images must be a path or list of paths.", sample_index)
    return images


def _normalize_modalities(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise AssetValidationError("MODEL_MODALITIES_INVALID", "modalities", "Model modalities must be a string list.")
    return sorted({item.lower() for item in value})


def _model_compatibility(modalities: Sequence[str]) -> dict[str, Any]:
    modes = []
    if "text" in modalities:
        modes.append("text_sft")
    if "text" in modalities and "image" in modalities:
        modes.append("image_text_sft")
    return {
        "mvp_compatible": bool(modes),
        "modes": modes,
        "reason": None if modes else "Model metadata does not declare text or image+text SFT input.",
    }


def _normalize_weight_precision(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower().replace("torch.", "")
    aliases = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"fp16", "bf16", "fp32", "int8", "int4"}:
        raise AssetValidationError(
            "MODEL_METADATA_INVALID", "weight_precision", "Model precision metadata is not recognized."
        )
    return normalized


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AssetValidationError("MODEL_METADATA_INVALID", field_name, f"{field_name} must be a positive integer.")
    return value


def _looks_like_secret(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return lowered.startswith(("hf_", "ms_token_", "modelscope_token_")) or len(value) > 128


def _number_stats(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "average": None}
    return {"min": min(values), "max": max(values), "average": fmean(values)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise RuntimeError(f"Persistent MVP asset index is invalid: {path}") from err
    if not isinstance(payload, dict):
        raise RuntimeError(f"Persistent MVP asset index must be an object: {path}")
    return payload


def _write_json_atomic(path: Path | None, payload: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    temporary_path.replace(path)
