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

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from sft_platform.api.jobs import create_training_app
from sft_platform.mvp.jobs import TrainingJobService


class MvpTrainingJobApiContractTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.models = root / "models"
        self.datasets = root / "datasets"
        self.outputs = root / "outputs"
        for path in (self.models, self.datasets, self.outputs):
            path.mkdir()
        self.service = TrainingJobService(root / "jobs.sqlite3")
        self.client = TestClient(create_training_app(self.service))
        registered = self.client.post(
            "/api/v1/compute-nodes/register",
            json={
                "name": "cloud-worker",
                "node_type": "cloud",
                "endpoint": "https://worker.example.test/agent",
                "model_root": str(self.models),
                "data_root": str(self.datasets),
                "output_root": str(self.outputs),
                "labels": {"region": "fake-cloud"},
            },
        )
        assert registered.status_code == 201
        self.node_id = registered.json()["id"]
        heartbeat = self.client.post(
            f"/api/v1/compute-nodes/{self.node_id}/heartbeat",
            json={
                "gpus": [
                    {
                        "id": 0,
                        "model": "Fake GPU",
                        "memory_total_bytes": 24_000,
                        "memory_free_bytes": 20_000,
                        "utilization": 0.0,
                    }
                ],
                "disk_free_bytes": 1_000_000,
                "driver_version": "fake-driver",
            },
        )
        assert heartbeat.status_code == 200

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def create_job(self):
        response = self.client.post(
            "/api/v1/training-jobs",
            json={
                "name": "text lora",
                "model_id": "model-api-fake",
                "dataset_snapshot_id": "dataset-api-fake",
                "training_parameters": {
                    "stage": "sft",
                    "finetuning_type": "lora",
                    "template": "qwen",
                },
                "gpu_count": 1,
                "node_id": self.node_id,
            },
        )
        assert response.status_code == 201, response.text
        return response.json()

    def test_full_create_query_agent_log_artifact_stop_and_resume_contract(self):
        job = self.create_job()
        listed = self.client.get("/api/v1/training-jobs")
        assert listed.status_code == 200
        assert listed.json()["items"][0]["id"] == job["id"]
        assert self.client.get(f"/api/v1/training-jobs/{job['id']}").json()["state"] == "QUEUED"

        polled = self.client.post("/api/v1/agent/tasks/poll", json={"node_id": self.node_id})
        assert polled.status_code == 200
        assignment = polled.json()["assignment"]
        attempt_id = assignment["attempt_id"]
        assert assignment["argv"] == ["sft-train", "train", "<controlled-job.yaml>"]
        assert "shell" not in assignment

        running = self.client.post(
            f"/api/v1/agent/tasks/{attempt_id}/state",
            json={
                "node_id": self.node_id,
                "state": "RUNNING",
                "metrics": {"loss": 1.0, "progress": 0.5, "gpu_memory_bytes": None},
            },
        )
        assert running.status_code == 200
        output_dir = Path(running.json()["output_dir"])
        logs = self.client.post(
            f"/api/v1/agent/tasks/{attempt_id}/logs",
            json={"node_id": self.node_id, "entries": [{"stream": "stdout", "message": "step 1"}]},
        )
        assert logs.status_code == 200
        artifacts = self.client.post(
            f"/api/v1/agent/tasks/{attempt_id}/artifacts",
            json={
                "node_id": self.node_id,
                "artifacts": [
                    {
                        "kind": "checkpoint",
                        "path": str(output_dir / "checkpoint-fake"),
                        "bytes": 0,
                        "metadata": {"fake": True},
                        "restorable": True,
                    }
                ],
            },
        )
        assert artifacts.status_code == 200
        checkpoint_id = artifacts.json()["items"][0]["id"]

        stopping = self.client.post(f"/api/v1/training-jobs/{job['id']}/stop")
        assert stopping.status_code == 200
        assert stopping.json()["state"] == "STOPPING"
        stopped = self.client.post(
            f"/api/v1/agent/tasks/{attempt_id}/state",
            json={"node_id": self.node_id, "state": "STOPPED"},
        )
        assert stopped.status_code == 200
        assert stopped.json()["state"] == "STOPPED"

        events = self.client.get(f"/api/v1/training-jobs/{job['id']}/events")
        assert events.status_code == 200
        assert events.json()["items"][-1]["state"] == "STOPPED"
        log_response = self.client.get(f"/api/v1/training-jobs/{job['id']}/logs")
        assert log_response.json()["items"][0]["message"] == "step 1"
        downloaded = self.client.get(f"/api/v1/training-jobs/{job['id']}/logs?download=true")
        assert downloaded.status_code == 200
        assert downloaded.headers["content-disposition"].endswith(f'"{job["id"]}.log"')
        assert "step 1" in downloaded.text
        artifact_list = self.client.get(f"/api/v1/training-jobs/{job['id']}/artifacts")
        assert artifact_list.json()["items"][0]["id"] == checkpoint_id

        resumed = self.client.post(
            f"/api/v1/training-jobs/{job['id']}/resume", json={"checkpoint_id": checkpoint_id}
        )
        assert resumed.status_code == 201
        assert resumed.json()["parent_job_id"] == job["id"]
        assert resumed.json()["state"] == "QUEUED"

    def test_cpu_only_node_can_connect_but_cannot_accept_training(self):
        registered = self.client.post(
            "/api/v1/compute-nodes/register",
            json={
                "name": "cpu-only-worker",
                "node_type": "local",
                "endpoint": "https://cpu-worker.example.test/agent",
                "model_root": str(self.models),
                "data_root": str(self.datasets),
                "output_root": str(self.outputs),
                "labels": {"purpose": "framework-validation"},
            },
        )
        assert registered.status_code == 201
        cpu_node_id = registered.json()["id"]
        heartbeat = self.client.post(
            f"/api/v1/compute-nodes/{cpu_node_id}/heartbeat",
            json={
                "gpus": [],
                "disk_free_bytes": 1_000_000,
                "system_memory_free_bytes": 2_000_000,
                "driver_version": "cpu-only",
            },
        )
        assert heartbeat.status_code == 200
        assert heartbeat.json()["status"] == "ONLINE"
        assert heartbeat.json()["gpus"] == []

        rejected = self.client.post(
            "/api/v1/training-jobs",
            json={
                "name": "cannot run without gpu",
                "model_id": "model-api-fake",
                "dataset_snapshot_id": "dataset-api-fake",
                "training_parameters": {"stage": "sft", "finetuning_type": "lora", "template": "qwen"},
                "gpu_count": 1,
                "node_id": cpu_node_id,
            },
        )
        assert rejected.status_code == 409
        assert rejected.json()["detail"]["code"] == "GPU_RESOURCE_INSUFFICIENT"

    def test_node_contract_and_rejections_are_sanitized_and_stable(self):
        nodes = self.client.get("/api/v1/compute-nodes")
        assert nodes.status_code == 200
        node = nodes.json()["items"][0]
        assert node["type"] == "cloud"
        assert node["status"] == "ONLINE"
        assert node["gpus"][0]["model"] == "Fake GPU"

        illegal_stage = self.client.post(
            "/api/v1/training-jobs",
            json={
                "name": "illegal",
                "model_id": "model-api-fake",
                "dataset_snapshot_id": "dataset-api-fake",
                "training_parameters": {"stage": "dpo", "finetuning_type": "lora"},
                "node_id": self.node_id,
            },
        )
        assert illegal_stage.status_code == 422
        assert illegal_stage.json()["detail"]["code"] == "MVP_TRAINING_STAGE_NOT_SUPPORTED"

        shell = self.client.post(
            "/api/v1/training-jobs",
            json={
                "name": "shell",
                "model_id": "model-api-fake",
                "dataset_snapshot_id": "dataset-api-fake",
                "training_parameters": {"shell_command": "echo injected"},
                "node_id": self.node_id,
            },
        )
        assert shell.status_code == 422
        assert shell.json()["detail"]["code"] == "TRAINING_PARAMETER_NOT_ALLOWED"

        secret = self.client.post(
            "/api/v1/training-jobs",
            json={
                "name": "secret",
                "model_id": "model-api-fake",
                "dataset_snapshot_id": "dataset-api-fake",
                "training_parameters": {"template": "qwen"},
                "node_id": self.node_id,
                "token": "must-not-be-reflected",
            },
        )
        assert secret.status_code == 422
        assert "must-not-be-reflected" not in secret.text


if __name__ == "__main__":
    unittest.main()
