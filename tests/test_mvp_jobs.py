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

from sft_platform.mvp.jobs import (
    ComputeNodeRegistration,
    FakeWorkerAgent,
    GpuReport,
    JobServiceError,
    JobSubmission,
    TrainingJobService,
)


class ManualClock:
    def __init__(self):
        self.value = 1_800_000_000.0

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class DeterministicIds:
    def __init__(self):
        self.next_id = 0

    def __call__(self, prefix):
        self.next_id += 1
        return f"{prefix}-{self.next_id:04d}"


class MvpTrainingJobServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_root = self.root / "models"
        self.data_root = self.root / "datasets"
        self.output_root = self.root / "outputs"
        for path in (self.model_root, self.data_root, self.output_root):
            path.mkdir()
        self.clock = ManualClock()
        self.ids = DeterministicIds()
        self.database = self.root / "service.sqlite3"
        self.service = TrainingJobService(
            self.database,
            clock=self.clock,
            id_generator=self.ids,
            lease_ttl_seconds=30,
            heartbeat_timeout_seconds=45,
            max_log_chunk_bytes=256,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def register_node(self, node_type="local", name="worker", gpu_count=2, disk_free_bytes=10_000):
        node = self.service.register_node(
            ComputeNodeRegistration(
                name=name,
                node_type=node_type,
                endpoint=f"https://{name}.example.test/agent",
                model_root=str(self.model_root),
                data_root=str(self.data_root),
                output_root=str(self.output_root),
                labels={"region": "test", "type": node_type},
            )
        )
        self.service.heartbeat(
            node["id"],
            [GpuReport(index, "Fake GPU", 24_000, 20_000, 0.1) for index in range(gpu_count)],
            disk_free_bytes=disk_free_bytes,
            driver_version="fake-driver-1",
        )
        return self.service.get_node(node["id"])

    def submit(self, *, node_id=None, gpu_count=1, minimum_disk_bytes=0, parameters=None):
        return self.service.create_job(
            JobSubmission(
                name="image-text qlora",
                model_id="model-fake-image-text",
                dataset_snapshot_id="dataset-fake-v1",
                training_parameters=parameters
                or {
                    "stage": "sft",
                    "finetuning_type": "lora",
                    "template": "qwen3_vl",
                    "quantization_bit": 4,
                    "quantization_method": "bnb",
                },
                gpu_count=gpu_count,
                node_id=node_id,
                minimum_disk_bytes=minimum_disk_bytes,
            )
        )

    def assert_service_error(self, code, callback):
        with self.assertRaises(JobServiceError) as context:
            callback()
        assert context.exception.code == code
        return context.exception

    def test_local_and_cloud_nodes_share_inventory_and_fake_execution_contract(self):
        for node_type in ("local", "cloud"):
            with self.subTest(node_type=node_type):
                node = self.register_node(node_type=node_type, name=f"{node_type}-node", gpu_count=2)
                assert node["type"] == node_type
                assert node["status"] == "ONLINE"
                assert node["driver_version"] == "fake-driver-1"
                assert len(node["gpus"]) == 2
                assert node["gpus"][0]["memory_free_bytes"] == 20_000
                job = self.submit(node_id=node["id"], gpu_count=2)
                completed = FakeWorkerAgent(self.service, node["id"]).run_next()
                assert completed["state"] == "SUCCEEDED"
                assert completed["attempts"][0]["allocated_gpu_ids"] == [0, 1]
                assert {item["kind"] for item in self.service.list_artifacts(job["id"])} == {
                    "adapter",
                    "checkpoint",
                    "log",
                }

    def test_persistent_queue_and_gpu_leases_prevent_duplicate_allocation(self):
        node = self.register_node(gpu_count=2)
        first = self.submit(node_id=node["id"], gpu_count=2)
        second = self.submit(node_id=node["id"], gpu_count=1)
        assignment = self.service.poll_assignment(node["id"])
        assert assignment["job_id"] == first["id"]
        assert assignment["gpu_ids"] == [0, 1]
        assert self.service.poll_assignment(node["id"]) is None
        assert all(gpu["lease"] is not None for gpu in self.service.get_node(node["id"])["gpus"])

        self.service.report_state(node["id"], assignment["attempt_id"], "RUNNING")
        self.service.report_state(node["id"], assignment["attempt_id"], "SUCCEEDED")
        next_assignment = self.service.poll_assignment(node["id"])
        assert next_assignment["job_id"] == second["id"]
        assert next_assignment["gpu_ids"] == [0]

    def test_state_machine_rejects_invalid_transitions_and_stop_converges(self):
        node = self.register_node(gpu_count=1)
        queued = self.submit(node_id=node["id"])
        stopped = self.service.stop_job(queued["id"])
        assert stopped["state"] == "STOPPED"
        assert [event["state"] for event in self.service.get_events(queued["id"])["items"] if event["type"] == "STATE_CHANGED"] == [
            "VALIDATING",
            "QUEUED",
            "STOPPING",
            "STOPPED",
        ]

        running = self.submit(node_id=node["id"])
        assignment = self.service.poll_assignment(node["id"])
        self.assert_service_error(
            "INVALID_STATE_TRANSITION",
            lambda: self.service.report_state(node["id"], assignment["attempt_id"], "SUCCEEDED"),
        )
        self.service.report_state(node["id"], assignment["attempt_id"], "RUNNING")
        stopping = self.service.stop_job(running["id"])
        assert stopping["state"] == "STOPPING"
        heartbeat = self.service.heartbeat(
            node["id"],
            [GpuReport(0, "Fake GPU", 24_000, 20_000)],
            disk_free_bytes=10_000,
            driver_version="fake-driver-1",
            active_attempt_ids=[assignment["attempt_id"]],
        )
        assert heartbeat["commands"] == [
            {"type": "STOP", "attempt_id": assignment["attempt_id"], "job_id": running["id"]}
        ]
        assert self.service.report_state(node["id"], assignment["attempt_id"], "STOPPED")["state"] == "STOPPED"

    def test_resume_creates_a_new_lineage_job_from_controlled_checkpoint(self):
        node = self.register_node(gpu_count=1)
        original = self.submit(node_id=node["id"])
        completed = FakeWorkerAgent(self.service, node["id"]).run_next()
        checkpoint = next(item for item in self.service.list_artifacts(original["id"]) if item["kind"] == "checkpoint")
        resumed = self.service.resume_job(completed["id"], checkpoint["id"])
        assert completed["state"] == "SUCCEEDED"
        assert resumed["state"] == "QUEUED"
        assert resumed["id"] != completed["id"]
        assert resumed["parent_job_id"] == completed["id"]
        assert resumed["parent_checkpoint_id"] == checkpoint["id"]
        assignment = self.service.poll_assignment(node["id"])
        assert assignment["config"]["resume_from_checkpoint"] == checkpoint["path"]

    def test_restart_preserves_jobs_events_logs_artifacts_and_reconciles_agent_report(self):
        node = self.register_node(gpu_count=1)
        job = self.submit(node_id=node["id"])
        assignment = self.service.poll_assignment(node["id"])
        self.service.report_state(node["id"], assignment["attempt_id"], "RUNNING", metrics={"progress": 0.5})
        self.service.append_logs(
            node["id"], assignment["attempt_id"], [{"stream": "stdout", "message": "before restart"}]
        )

        restarted = TrainingJobService(
            self.database,
            clock=self.clock,
            id_generator=self.ids,
            lease_ttl_seconds=30,
            heartbeat_timeout_seconds=45,
        )
        restored_job = restarted.get_job(job["id"])
        assert restored_job["state"] == "RUNNING"
        assert restored_job["code_version"] == "4451765a6b04ff08a6c5650f5953513608ae9e64"
        assert restarted.get_logs(job["id"])["items"][0]["message"] == "before restart"
        restarted.heartbeat(
            node["id"],
            [GpuReport(0, "Fake GPU", 24_000, 20_000)],
            disk_free_bytes=10_000,
            driver_version="fake-driver-1",
            active_attempt_ids=[assignment["attempt_id"]],
        )
        assert restarted.report_state(node["id"], assignment["attempt_id"], "SUCCEEDED")["state"] == "SUCCEEDED"
        assert restarted.get_events(job["id"])["items"][-1]["state"] == "SUCCEEDED"

    def test_expired_lease_fails_in_place_before_gpu_can_be_reallocated(self):
        node = self.register_node(gpu_count=1)
        first = self.submit(node_id=node["id"])
        assignment = self.service.poll_assignment(node["id"])
        self.service.report_state(node["id"], assignment["attempt_id"], "RUNNING")
        second = self.submit(node_id=node["id"])
        self.clock.advance(31)
        self.service.reconcile()
        failed = self.service.get_job(first["id"])
        assert failed["state"] == "FAILED"
        assert failed["error"]["category"] == "NODE_DISCONNECTED"
        assert failed["error"]["code"] == "GPU_LEASE_EXPIRED"
        assert self.service.poll_assignment(node["id"])["job_id"] == second["id"]

    def test_registered_asset_references_and_modalities_are_checked_when_composed(self):
        class Models:
            def get(self, model_id):
                if model_id != "model-known":
                    raise KeyError(model_id)
                return {"compatibility": {"mvp_compatible": True, "modes": ["text_sft"]}}

        class Datasets:
            def get_snapshot(self, snapshot_id):
                if snapshot_id != "dataset-known":
                    raise KeyError(snapshot_id)
                return [{"messages": [], "images": ["fake.png"]}]

        composed = TrainingJobService(
            self.root / "composed.sqlite3",
            clock=self.clock,
            id_generator=self.ids,
            model_service=Models(),
            dataset_service=Datasets(),
        )
        submission = JobSubmission("checked", "model-known", "dataset-known", {"template": "qwen"})
        with self.assertRaises(JobServiceError) as context:
            composed.create_job(submission)
        assert context.exception.code == "MODEL_DATASET_MODALITY_MISMATCH"

    def test_resources_paths_shell_fields_and_secrets_are_rejected(self):
        node = self.register_node(gpu_count=1, disk_free_bytes=10)
        self.assert_service_error(
            "DISK_INSUFFICIENT", lambda: self.submit(node_id=node["id"], minimum_disk_bytes=11)
        )
        self.assert_service_error(
            "GPU_RESOURCE_INSUFFICIENT", lambda: self.submit(node_id=node["id"], gpu_count=2)
        )
        self.assert_service_error(
            "MANAGED_PARAMETER_NOT_ALLOWED",
            lambda: self.submit(node_id=node["id"], parameters={"output_dir": "/tmp/injected"}),
        )
        self.assert_service_error(
            "TRAINING_PARAMETER_NOT_ALLOWED",
            lambda: self.submit(node_id=node["id"], parameters={"shell_command": "rm -rf /"}),
        )
        self.assert_service_error(
            "PLAINTEXT_CREDENTIAL_NOT_ALLOWED",
            lambda: self.submit(node_id=node["id"], parameters={"template": "hf_plain_secret"}),
        )
        job = self.submit(node_id=node["id"])
        assignment = self.service.poll_assignment(node["id"])
        self.assert_service_error(
            "PATH_OUTSIDE_CONTROLLED_ROOT",
            lambda: self.service.add_artifacts(
                node["id"], assignment["attempt_id"], [{"kind": "checkpoint", "path": "/tmp/escape"}]
            ),
        )
        assert job["command_preview"] == ["sft-train", "train", "<controlled-job.yaml>"]

    def test_logs_are_redacted_rate_limited_and_failures_are_classified(self):
        node = self.register_node(gpu_count=1)
        job = self.submit(node_id=node["id"])
        assignment = self.service.poll_assignment(node["id"])
        self.service.report_state(node["id"], assignment["attempt_id"], "RUNNING")
        self.service.append_logs(
            node["id"],
            assignment["attempt_id"],
            [{"stream": "stderr", "message": "authorization Bearer private-value and hf_secretvalue"}],
        )
        assert "private-value" not in self.service.get_logs(job["id"])["items"][0]["message"]
        self.assert_service_error(
            "LOG_RATE_LIMIT_EXCEEDED",
            lambda: self.service.append_logs(
                node["id"], assignment["attempt_id"], [{"stream": "stdout", "message": "x" * 300}]
            ),
        )
        failed = self.service.report_state(
            node["id"],
            assignment["attempt_id"],
            "FAILED",
            error_category="NETWORK",
            error_code="MODEL_HUB_UNREACHABLE",
            error_message="network failed with hf_private",
        )
        assert failed["error"]["category"] == "NETWORK"
        assert "hf_private" not in failed["error"]["message"]


if __name__ == "__main__":
    unittest.main()
