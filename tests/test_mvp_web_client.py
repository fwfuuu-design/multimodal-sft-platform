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

import httpx
import pytest

from sft_platform.webui.mvp_client import (
    MvpWebApiClient,
    MvpWebApiError,
    WebRequestContext,
    compute_node_view,
)


def test_request_context_requires_oidc_bearer_and_forwards_project():
    with pytest.raises(MvpWebApiError) as error:
        WebRequestContext.from_headers({"X-Project-ID": "project-a"})
    assert error.value.code == "AUTH_REQUIRED"

    context = WebRequestContext.from_headers(
        {"Authorization": "Bearer fake-opaque-token", "X-Project-ID": "project-a"}
    )
    assert context.api_headers() == {
        "Authorization": "Bearer fake-opaque-token",
        "Accept": "application/json",
        "X-Project-ID": "project-a",
    }


def test_client_reads_identity_and_cpu_gpu_nodes_without_leaking_token():
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        if request.url.path == "/api/v1/auth/me":
            return httpx.Response(
                200,
                json={"user_id": "user-1", "display_name": "Tester", "role": "trainer", "project_id": "project-a"},
            )
        if request.url.path == "/api/v1/compute-nodes":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": "node-cpu", "name": "CPU node", "type": "local", "status": "ONLINE", "gpus": []},
                        {
                            "id": "node-gpu",
                            "name": "GPU node",
                            "type": "cloud",
                            "status": "ONLINE",
                            "gpus": [{"id": 0, "model": "Fake GPU"}],
                        },
                    ]
                },
            )
        return httpx.Response(404)

    context = WebRequestContext.from_headers(
        {"authorization": "Bearer fake-opaque-token", "x-project-id": "project-a"}
    )
    with MvpWebApiClient("https://api.example.test", transport=httpx.MockTransport(handler)) as client:
        identity = client.get_identity(context)
        nodes = client.list_compute_nodes(context)

    assert identity["role"] == "trainer"
    assert all(request.headers["authorization"] == "Bearer fake-opaque-token" for request in seen_requests)
    assert all(request.headers["x-project-id"] == "project-a" for request in seen_requests)
    assert "fake-opaque-token" not in repr(identity) + repr(nodes)

    choices, summary = compute_node_view(nodes)
    assert choices == [
        ("CPU node · 本地 · 在线 · 无 GPU（仅接入）", "node-cpu"),
        ("GPU node · 云端 · 在线 · 1 GPU", "node-gpu"),
    ]
    assert summary == "节点 2 个 · 在线 2 个 · 可训练 1 个 · 无 GPU 1 个"


def test_client_converts_transport_and_api_failures_to_safe_messages():
    def unavailable(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret-host failed")

    context = WebRequestContext("Bearer should-not-be-shown")
    with MvpWebApiClient("https://api.example.test", transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(MvpWebApiError) as error:
            client.list_compute_nodes(context)
    assert error.value.code == "API_UNAVAILABLE"
    assert "secret-host" not in error.value.message
    assert "should-not-be-shown" not in error.value.message

    def forbidden(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": {"code": "ROLE_READ_ONLY", "message": "只读角色无权操作。"}})

    with MvpWebApiClient("https://api.example.test", transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(MvpWebApiError) as error:
            client.get_identity(context)
    assert error.value.code == "ROLE_READ_ONLY"
    assert error.value.status_code == 403
