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

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from sft_platform.mvp.operations import (
    DATABASE_SCHEMA_VERSION,
    OperationsError,
    build_backup_plan,
    build_restore_plan,
    load_restore_manifest,
    prepare_database_schema,
    run_backup_plan_cli,
    run_restore_plan_cli,
)


class TestMvpOperations:
    def setup_method(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("state", "data", "media", "results", "worker-output"):
            (self.root / name).mkdir()
        self.database = self.root / "state" / "service.sqlite3"
        prepare_database_schema(self.database)
        self.environment = {
            "MVP_DATABASE_PATH": str(self.database),
            "MVP_STATE_ROOT": str(self.root / "state"),
            "MVP_DATA_ROOTS": str(self.root / "data"),
            "MVP_MEDIA_ROOTS": str(self.root / "media"),
            "MVP_RESULTS_ROOT": str(self.root / "results"),
            "MVP_FERNET_KEY": "must-never-appear-in-a-plan",
        }

    def teardown_method(self):
        self.temporary.cleanup()

    def test_schema_version_initializes_and_rejects_newer_or_unplanned_older_database(self):
        assert prepare_database_schema(self.database) == DATABASE_SCHEMA_VERSION
        with sqlite3.connect(self.database) as connection:
            assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
            connection.execute("UPDATE schema_version SET version = 2")
        with pytest.raises(OperationsError, match="DATABASE_SCHEMA_TOO_NEW"):
            prepare_database_schema(self.database)

        older = self.root / "state" / "older.sqlite3"
        prepare_database_schema(older)
        with sqlite3.connect(older) as connection:
            connection.execute("UPDATE schema_version SET version = 0")
        with pytest.raises(OperationsError, match="DATABASE_MIGRATION_REQUIRED"):
            prepare_database_schema(older)

    def test_backup_and_restore_plans_are_non_destructive_and_include_worker_outputs(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE nodes (output_root TEXT NOT NULL)")
            connection.execute("INSERT INTO nodes VALUES (?)", (str(self.root / "worker-output"),))
        plan = build_backup_plan(self.environment, generated_at="2026-09-03T00:00:00+00:00")
        serialized = json.dumps(plan)
        assert plan["dry_run"] is True
        assert plan["physical_changes"] is False
        assert plan["database_schema_version"] == DATABASE_SCHEMA_VERSION
        assert plan["recovery_policy"] == {
            "rpo_hours": 24,
            "rto_hours": 4,
            "daily_backup_retention_days": 30,
            "monthly_backup_retention_months": 6,
            "recovery_exercise_interval_months": 3,
        }
        assert {source["kind"] for source in plan["sources"]} >= {
            "database",
            "state",
            "data",
            "media",
            "results",
            "worker_output",
        }
        assert "must-never-appear-in-a-plan" not in serialized

        restore = build_restore_plan(plan)
        assert restore["dry_run"] is True
        assert restore["physical_changes"] is False
        assert "confirm the target environment and overwrite scope" in restore["steps"]

    def test_restore_rejects_invalid_or_incompatible_manifests(self):
        plan = build_backup_plan(self.environment, generated_at="2026-09-03T00:00:00+00:00")
        with pytest.raises(OperationsError, match="RESTORE_MANIFEST_INVALID"):
            build_restore_plan({})
        with pytest.raises(OperationsError, match="RESTORE_CODE_VERSION_MISMATCH"):
            build_restore_plan({**plan, "code_version": "wrong"})
        with pytest.raises(OperationsError, match="RESTORE_SCHEMA_VERSION_MISMATCH"):
            build_restore_plan({**plan, "database_schema_version": 999})

        invalid = self.root / "invalid.json"
        invalid.write_text("[]", encoding="utf-8")
        with pytest.raises(OperationsError, match="RESTORE_MANIFEST_INVALID"):
            load_restore_manifest(invalid)

    def test_cli_operations_require_explicit_dry_run_and_only_print_plans(self, monkeypatch, capsys):
        for key, value in self.environment.items():
            monkeypatch.setenv(key, value)
        with pytest.raises(OperationsError, match="BACKUP_DRY_RUN_REQUIRED"):
            run_backup_plan_cli([])
        run_backup_plan_cli(["--dry-run"])
        backup = json.loads(capsys.readouterr().out)

        manifest_path = self.root / "backup-plan.json"
        manifest_path.write_text(json.dumps(backup), encoding="utf-8")
        with pytest.raises(OperationsError, match="RESTORE_DRY_RUN_REQUIRED"):
            run_restore_plan_cli([str(manifest_path)])
        run_restore_plan_cli(["--dry-run", str(manifest_path)])
        restore = json.loads(capsys.readouterr().out)
        assert restore["kind"] == "mvp-restore-plan"
        assert restore["physical_changes"] is False
