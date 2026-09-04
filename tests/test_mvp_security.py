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

import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from sft_platform.api.jobs import create_secure_mvp_app
from sft_platform.mvp.assets import DatasetAssetService, FakeHub, ModelAssetService
from sft_platform.mvp.jobs import ERROR_CATEGORIES, FakeWorkerAgent, JobSubmission, TrainingJobService
from sft_platform.mvp.results import ResultService
from sft_platform.mvp.security import FakeOIDCVerifier, Identity, SecurityError, SecurityService


class ManualClock:
    def __init__(self):
        self.value = 1_800_000_000.0

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class DeterministicIds:
    def __init__(self):
        self.value = 0

    def __call__(self, prefix):
        self.value += 1
        return f"{prefix}-{self.value:04d}"


class MvpSecurityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.models_root = self.root / "models"
        self.datasets_root = self.root / "datasets"
        self.media_root = self.root / "media"
        self.outputs_root = self.root / "outputs"
        self.results_root = self.root / "results"
        for path in (
            self.models_root,
            self.datasets_root,
            self.media_root,
            self.outputs_root,
            self.results_root,
        ):
            path.mkdir()
        self.database = self.root / "service.sqlite3"
        self.clock = ManualClock()
        self.ids = DeterministicIds()
        self.model_service = ModelAssetService(
            [self.models_root],
            FakeHub(
                {
                    ("huggingface", "company/secure-model"): {
                        "private": True,
                        "parameter_count": 1_000,
                        "estimated_model_bytes": 2_000,
                        "weight_precision": "fp16",
                        "template": "qwen",
                        "context_length": 8192,
                        "modalities": ["text"],
                    }
                }
            ),
        )
        self.dataset_service = DatasetAssetService([self.datasets_root], [self.media_root])
        self.training_service = TrainingJobService(
            self.database, model_service=self.model_service, dataset_service=self.dataset_service
        )
        self.result_service = ResultService(
            self.database,
            self.results_root,
            model_service=self.model_service,
            dataset_service=self.dataset_service,
            training_service=self.training_service,
        )
        verifier = FakeOIDCVerifier(
            {
                "admin-token": Identity("admin", "Administrator"),
                "alice-token": Identity("alice", "Alice"),
                "charlie-token": Identity("charlie", "Charlie"),
                "viewer-token": Identity("viewer", "Viewer"),
                "bob-token": Identity("bob", "Bob"),
                "agent-token": Identity("worker-agent", "Worker Agent"),
            }
        )
        self.security = SecurityService(
            self.database,
            token_verifier=verifier,
            encryption_key=Fernet.generate_key(),
            training_service=self.training_service,
            clock=self.clock,
            id_generator=self.ids,
        )
        self.project = self.security.create_project("Project One", "admin", "Administrator")["id"]
        self.security.add_member(self.project, "alice", "Alice", "trainer")
        self.security.add_member(self.project, "charlie", "Charlie", "trainer")
        self.security.add_member(self.project, "viewer", "Viewer", "viewer")
        self.security.add_member(self.project, "worker-agent", "Worker Agent", "agent")
        self.other_project = self.security.create_project("Project Two", "bob", "Bob")["id"]
        self.security.add_member(self.other_project, "bob", "Bob", "trainer")
        self.client = TestClient(
            create_secure_mvp_app(
                self.model_service,
                self.dataset_service,
                self.training_service,
                self.result_service,
                self.security,
            )
        )
        self.admin = self.headers("admin-token", self.project)
        self.alice = self.headers("alice-token", self.project)
        self.charlie = self.headers("charlie-token", self.project)
        self.viewer = self.headers("viewer-token", self.project)
        self.agent = self.headers("agent-token", self.project)
        self.bob = self.headers("bob-token", self.other_project)
        self.admin_principal = self.security.authenticate("Bearer admin-token", self.project)
        credential = self.security.provision_credential(
            self.admin_principal, kind="huggingface", plaintext="hf_server_side_private_value"
        )
        self.credential_id = credential["id"]
        self.node_id = self.create_node()
        self.model_id = self.create_model()
        self.dataset_id = self.create_dataset()

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    @staticmethod
    def headers(token, project):
        return {"Authorization": f"Bearer {token}", "X-Project-ID": project}

    def create_node(self):
        response = self.client.post(
            "/api/v1/compute-nodes/register",
            headers=self.admin,
            json={
                "name": "secure-node",
                "node_type": "local",
                "endpoint": "https://worker.example.test/agent",
                "model_root": str(self.models_root),
                "data_root": str(self.datasets_root),
                "output_root": str(self.outputs_root),
                "labels": {},
            },
        )
        assert response.status_code == 201, response.text
        node_id = response.json()["id"]
        heartbeat = self.client.post(
            f"/api/v1/compute-nodes/{node_id}/heartbeat",
            headers=self.admin,
            json={
                "gpus": [
                    {
                        "id": 0,
                        "model": "Fake GPU",
                        "memory_total_bytes": 100_000,
                        "memory_free_bytes": 100_000,
                    }
                ],
                "disk_free_bytes": 100_000,
                "system_memory_free_bytes": 100_000,
                "driver_version": "fake-driver",
            },
        )
        assert heartbeat.status_code == 200, heartbeat.text
        return node_id

    def create_model(self):
        response = self.client.post(
            "/api/v1/models",
            headers=self.alice,
            json={
                "name": "secure-model",
                "source": "huggingface",
                "location": "company/secure-model",
                "credential_id": self.credential_id,
            },
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    def create_dataset(self):
        payload = json.dumps([{"instruction": "Hello", "input": "", "output": "World"}]).encode()
        uploaded = self.client.post(
            "/api/v1/datasets/upload",
            headers=self.alice,
            files={"file": ("train.json", io.BytesIO(payload), "application/json")},
        )
        assert uploaded.status_code == 201, uploaded.text
        inspected = self.client.post(
            "/api/v1/datasets/inspect",
            headers=self.alice,
            json={"dataset_file_id": uploaded.json()["id"], "formatting": "alpaca"},
        )
        assert inspected.status_code == 200, inspected.text
        return inspected.json()["snapshot_id"]

    def create_job(self, headers=None):
        response = self.client.post(
            "/api/v1/training-jobs",
            headers=headers or self.alice,
            json={
                "name": "secure job",
                "model_id": self.model_id,
                "dataset_snapshot_id": self.dataset_id,
                "training_parameters": {"stage": "sft", "finetuning_type": "lora", "template": "qwen"},
                "gpu_count": 1,
                "node_id": self.node_id,
            },
        )
        return response

    def complete_job(self):
        job_response = self.create_job()
        assert job_response.status_code == 201, job_response.text
        job = job_response.json()
        completed = FakeWorkerAgent(self.training_service, self.node_id).run_next()
        assert completed["state"] == "SUCCEEDED"
        adapter = next(
            item for item in self.training_service.list_artifacts(job["id"]) if item["kind"] == "adapter"
        )
        checkpoint = next(
            item for item in self.training_service.list_artifacts(job["id"]) if item["kind"] == "checkpoint"
        )
        return job, adapter, checkpoint

    def test_oidc_roles_project_visibility_and_read_only_access(self):
        missing = self.client.get("/api/v1/models", headers={"X-Project-ID": self.project})
        assert missing.status_code == 401
        me = self.client.get("/api/v1/auth/me", headers=self.viewer)
        assert me.status_code == 200
        assert me.json()["role"] == "viewer"

        viewer_list = self.client.get("/api/v1/models", headers=self.viewer)
        assert [item["id"] for item in viewer_list.json()["items"]] == [self.model_id]
        viewer_create = self.client.post(
            "/api/v1/training-jobs",
            headers=self.viewer,
            json={},
        )
        assert viewer_create.status_code == 403
        assert viewer_create.json()["detail"]["code"] == "ROLE_READ_ONLY"

        other_list = self.client.get("/api/v1/models", headers=self.bob)
        assert other_list.json()["items"] == []
        denied = self.client.get(f"/api/v1/models/{self.model_id}", headers=self.bob)
        assert denied.status_code == 404

        company = self.client.put(
            f"/api/v1/security/objects/model/{self.model_id}/visibility",
            headers=self.admin,
            json={"visibility": "company"},
        )
        assert company.status_code == 200
        assert self.client.get(f"/api/v1/models/{self.model_id}", headers=self.bob).status_code == 200

        private = self.client.put(
            f"/api/v1/security/objects/model/{self.model_id}/visibility",
            headers=self.admin,
            json={"visibility": "private"},
        )
        assert private.status_code == 200
        assert self.client.get(f"/api/v1/models/{self.model_id}", headers=self.charlie).status_code == 404
        assert self.client.get(f"/api/v1/models/{self.model_id}", headers=self.alice).status_code == 200

    def test_agent_identity_is_limited_to_worker_endpoints(self):
        heartbeat = self.client.post(
            f"/api/v1/compute-nodes/{self.node_id}/heartbeat",
            headers=self.agent,
            json={
                "gpus": [{"id": 0, "model": "Fake GPU", "memory_total_bytes": 100_000, "memory_free_bytes": 100_000}],
                "disk_free_bytes": 100_000,
                "system_memory_free_bytes": 100_000,
                "driver_version": "fake-driver",
            },
        )
        assert heartbeat.status_code == 200
        denied = self.client.get("/api/v1/models", headers=self.agent)
        assert denied.status_code == 403
        assert denied.json()["detail"]["code"] == "AGENT_SCOPE_DENIED"

    def test_only_admin_adds_nodes_and_trainers_cannot_stop_others_jobs(self):
        denied_node = self.client.post(
            "/api/v1/compute-nodes/register", headers=self.alice, json={}
        )
        assert denied_node.status_code == 403
        assert denied_node.json()["detail"]["code"] == "ADMIN_REQUIRED"

        job = self.create_job().json()
        assignment = self.training_service.poll_assignment(self.node_id)
        self.training_service.report_state(self.node_id, assignment["attempt_id"], "RUNNING")
        denied_stop = self.client.post(f"/api/v1/training-jobs/{job['id']}/stop", headers=self.charlie)
        assert denied_stop.status_code == 404
        stopped = self.client.post(f"/api/v1/training-jobs/{job['id']}/stop", headers=self.admin)
        assert stopped.status_code == 200
        assert stopped.json()["state"] == "STOPPING"

    def test_artifact_access_delete_and_sensitive_audit_are_enforced(self):
        job, adapter, checkpoint = self.complete_job()
        downloaded = self.client.get(
            f"/api/v1/training-jobs/{job['id']}/artifacts/{adapter['id']}/download",
            headers=self.alice,
        )
        assert downloaded.status_code == 200
        downloaded_log = self.client.get(
            f"/api/v1/training-jobs/{job['id']}/logs?download=true", headers=self.alice
        )
        assert downloaded_log.status_code == 200
        cross_project = self.client.get(
            f"/api/v1/training-jobs/{job['id']}/artifacts/{adapter['id']}/download",
            headers=self.bob,
        )
        assert cross_project.status_code == 404

        resumed = self.client.post(
            f"/api/v1/training-jobs/{job['id']}/resume",
            headers=self.alice,
            json={"checkpoint_id": checkpoint["id"]},
        )
        assert resumed.status_code == 201, resumed.text
        denied_delete = self.client.request(
            "DELETE",
            f"/api/v1/security/objects/dataset/{self.dataset_id}",
            headers=self.alice,
            json={"reason": "not allowed"},
        )
        assert denied_delete.status_code == 403
        deleted = self.client.request(
            "DELETE",
            f"/api/v1/security/objects/dataset/{self.dataset_id}",
            headers=self.admin,
            json={"reason": "approved cleanup"},
        )
        assert deleted.status_code == 200
        assert deleted.json() == {
            "object_type": "dataset",
            "object_id": self.dataset_id,
            "deleted": True,
            "physical_delete": False,
        }
        assert self.client.get(f"/api/v1/datasets/{self.dataset_id}/preview", headers=self.alice).status_code == 404

        audit = self.client.get("/api/v1/security/audit", headers=self.admin).json()["items"]
        actions = {(item["action"], item["outcome"]) for item in audit}
        assert ("training_job.create", "SUCCESS") in actions
        assert ("training_job.resume", "SUCCESS") in actions
        assert ("artifact.download", "SUCCESS") in actions
        assert ("log.download", "SUCCESS") in actions
        assert ("object.delete", "DENIED") in actions
        assert ("object.delete", "SUCCESS") in actions

    def test_credentials_are_encrypted_and_only_opaque_ids_are_exposed(self):
        credential_list = self.client.get("/api/v1/security/credentials", headers=self.alice)
        assert credential_list.status_code == 200
        assert credential_list.json()["items"][0]["credential_id"] == self.credential_id
        assert "hf_server_side_private_value" not in credential_list.text
        with sqlite3.connect(self.database) as connection:
            ciphertext = connection.execute(
                "SELECT ciphertext FROM security_credentials WHERE credential_id = ?", (self.credential_id,)
            ).fetchone()[0]
        assert b"hf_server_side_private_value" not in ciphertext
        assert self.security.resolve_credential(self.credential_id, self.project) == "hf_server_side_private_value"
        serialized_surfaces = json.dumps(
            {
                "job": self.create_job().json(),
                "models": self.client.get("/api/v1/models", headers=self.alice).json(),
            }
        )
        assert "hf_server_side_private_value" not in serialized_surfaces

    def test_job_upload_and_log_quotas_are_configurable(self):
        quota = self.client.put(
            "/api/v1/security/quotas",
            headers=self.admin,
            json={"user_id": "alice", "max_total_jobs": 1},
        )
        assert quota.status_code == 200
        assert self.create_job().status_code == 201
        exceeded = self.create_job()
        assert exceeded.status_code == 429
        assert exceeded.json()["detail"]["code"] == "JOB_TOTAL_QUOTA_EXCEEDED"

        self.security.set_quota(self.project, "alice", {"max_upload_bytes": 10})
        upload = self.client.post(
            "/api/v1/datasets/upload",
            headers=self.alice,
            files={"file": ("too-large.json", io.BytesIO(b"[]"), "application/json")},
        )
        assert upload.status_code == 413
        assert upload.json()["detail"]["code"] == "UPLOAD_QUOTA_EXCEEDED"

        self.security.set_quota(self.project, "admin", {"max_log_bytes_per_minute": 10})
        self.security.consume_log_quota(self.admin_principal, 6)
        with self.assertRaises(SecurityError) as context:
            self.security.consume_log_quota(self.admin_principal, 5)
        assert context.exception.code == "LOG_RATE_QUOTA_EXCEEDED"

    def test_concurrent_job_quota_blocks_a_second_active_job(self):
        self.security.set_quota(self.project, "alice", {"max_concurrent_jobs": 1})
        assert self.create_job().status_code == 201
        exceeded = self.create_job()
        assert exceeded.status_code == 429
        assert exceeded.json()["detail"]["code"] == "JOB_CONCURRENCY_QUOTA_EXCEEDED"

    def test_retention_soft_deletes_metadata_without_unconfirmed_physical_deletion(self):
        old_id = "dataset-old"
        self.security.bind_object(self.admin_principal, "dataset", old_id)
        self.clock.advance(2 * 86400)
        retained = self.client.post(
            "/api/v1/security/retention/run",
            headers=self.admin,
            json={"older_than_days": 1},
        )
        assert retained.status_code == 200
        assert {item["object_id"] for item in retained.json()["soft_deleted"]} >= {old_id}
        assert retained.json()["physical_delete"] is False
        assert self.security.object_record("dataset", old_id)["deleted_at"] is not None

    def test_audit_retention_keeps_the_configured_180_day_online_window(self):
        self.security.audit(
            self.admin_principal,
            action="old.event",
            resource_type="test",
            resource_id="old",
            outcome="SUCCESS",
            status_code=200,
        )
        self.clock.advance(181 * 86400)
        self.security.audit(
            self.admin_principal,
            action="new.event",
            resource_type="test",
            resource_id="new",
            outcome="SUCCESS",
            status_code=200,
        )
        assert self.security.prune_audit(self.admin_principal) >= 1
        actions = {item["action"] for item in self.security.list_audit(self.admin_principal)}
        assert "new.event" in actions
        assert "old.event" not in actions

    def test_all_approved_failure_categories_are_persisted(self):
        for category in sorted(ERROR_CATEGORIES):
            with self.subTest(category=category):
                job = self.training_service.create_job(
                    JobSubmission(
                        f"failure {category}",
                        self.model_id,
                        self.dataset_id,
                        {"stage": "sft", "finetuning_type": "lora", "template": "qwen"},
                        node_id=self.node_id,
                    )
                )
                assignment = self.training_service.poll_assignment(self.node_id)
                self.training_service.report_state(self.node_id, assignment["attempt_id"], "RUNNING")
                failed = self.training_service.report_state(
                    self.node_id,
                    assignment["attempt_id"],
                    "FAILED",
                    error_category=category,
                    error_code=f"FAKE_{category}",
                    error_message="classified fake failure",
                )
                assert failed["id"] == job["id"]
                assert failed["error"]["category"] == category

    def test_failed_job_keeps_configuration_logs_and_latest_checkpoint(self):
        response = self.create_job()
        job = response.json()
        assignment = self.training_service.poll_assignment(self.node_id)
        self.training_service.report_state(self.node_id, assignment["attempt_id"], "RUNNING")
        output = Path(assignment["config"]["output_dir"])
        output.mkdir(parents=True)
        checkpoint = output / "checkpoint-before-failure"
        checkpoint.mkdir()
        self.training_service.append_logs(
            self.node_id,
            assignment["attempt_id"],
            [{"stream": "stderr", "message": "controlled failure"}],
        )
        self.training_service.add_artifacts(
            self.node_id,
            assignment["attempt_id"],
            [{"kind": "checkpoint", "path": str(checkpoint), "metadata": {"step": 1}, "restorable": True}],
        )
        failed = self.training_service.report_state(
            self.node_id,
            assignment["attempt_id"],
            "FAILED",
            error_category="GPU_MEMORY",
            error_code="OUT_OF_MEMORY",
            error_message="fake OOM",
        )
        assert failed["config_snapshot"]["template"] == "qwen"
        assert failed["error"]["category"] == "GPU_MEMORY"
        assert self.training_service.get_logs(job["id"])["items"][0]["message"] == "controlled failure"
        assert self.training_service.list_artifacts(job["id"])[0]["restorable"] is True


if __name__ == "__main__":
    unittest.main()
