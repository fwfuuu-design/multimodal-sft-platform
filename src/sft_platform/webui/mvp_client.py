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
#
# Modified in 2026 for the multimodal fine-tuning product WebUI.

"""Fail-closed HTTP client and view helpers for the frozen MVP API v1."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class MvpWebApiError(RuntimeError):
    """Safe API error suitable for display without request credentials."""

    def __init__(self, code: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class WebRequestContext:
    authorization: str
    project_id: str | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> WebRequestContext:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        authorization = normalized.get("authorization", "")
        if not authorization.lower().startswith("bearer ") or not authorization[7:].strip():
            raise MvpWebApiError("AUTH_REQUIRED", "请先通过公司 SSO 登录。", 401)

        project_id = normalized.get("x-project-id") or None
        return cls(authorization=authorization, project_id=project_id)

    def api_headers(self) -> dict[str, str]:
        headers = {"Authorization": self.authorization, "Accept": "application/json"}
        if self.project_id:
            headers["X-Project-ID"] = self.project_id
        return headers


class MvpWebApiClient:
    """Small synchronous API v1 client used by Gradio callbacks."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        configured_url = base_url or os.getenv("MVP_API_BASE_URL", DEFAULT_API_BASE_URL)
        self._client = httpx.Client(
            base_url=configured_url.rstrip("/"), timeout=timeout_seconds, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MvpWebApiClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _get(self, path: str, context: WebRequestContext) -> dict[str, Any]:
        try:
            response = self._client.get(path, headers=context.api_headers())
        except httpx.RequestError as error:
            raise MvpWebApiError("API_UNAVAILABLE", "训练服务暂时不可用，请稍后重试。") from error

        if response.is_error:
            code = "API_REQUEST_FAILED"
            message = "训练服务请求失败。"
            try:
                detail = response.json().get("detail", {})
                if isinstance(detail, dict):
                    code = str(detail.get("code") or code)
                    message = str(detail.get("message") or message)
            except (TypeError, ValueError):
                pass
            raise MvpWebApiError(code, message, response.status_code)

        payload = response.json()
        if not isinstance(payload, dict):
            raise MvpWebApiError("API_RESPONSE_INVALID", "训练服务返回了无效响应。")
        return payload

    def get_identity(self, context: WebRequestContext) -> dict[str, Any]:
        return self._get("/api/v1/auth/me", context)

    def list_compute_nodes(self, context: WebRequestContext) -> list[dict[str, Any]]:
        items = self._get("/api/v1/compute-nodes", context).get("items", [])
        if not isinstance(items, list):
            raise MvpWebApiError("API_RESPONSE_INVALID", "计算节点响应格式无效。")
        return [item for item in items if isinstance(item, dict)]


def compute_node_view(nodes: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], str]:
    """Build selector choices and a concise resource summary, including CPU-only nodes."""
    choices: list[tuple[str, str]] = []
    online = 0
    trainable = 0
    cpu_only = 0
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        name = str(node.get("name") or node_id)
        node_type = "云端" if node.get("type") == "cloud" else "本地"
        status = str(node.get("status") or "UNKNOWN").upper()
        status_label = {"ONLINE": "在线", "OFFLINE": "离线", "ERROR": "异常"}.get(status, "未知")
        gpus = node.get("gpus") if isinstance(node.get("gpus"), list) else []
        gpu_count = len(gpus)
        if status == "ONLINE":
            online += 1
        if status == "ONLINE" and gpu_count:
            trainable += 1
        if gpu_count == 0:
            cpu_only += 1
        resource = f"{gpu_count} GPU" if gpu_count else "无 GPU（仅接入）"
        choices.append((f"{name} · {node_type} · {status_label} · {resource}", node_id))

    summary = f"节点 {len(choices)} 个 · 在线 {online} 个 · 可训练 {trainable} 个 · 无 GPU {cpu_only} 个"
    if not choices:
        summary = "尚未发现计算节点。无 GPU 服务器也可以注册，但不能领取训练任务。"
    return choices, summary
