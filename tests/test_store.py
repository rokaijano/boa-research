from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boaresearch.schema import CandidateMetadata, PatchDescriptor, StageRunResult, StageCommandResult
from boaresearch.store import ExperimentStore


class StoreTests(unittest.TestCase):
    def test_store_round_trip(self) -> None:
        db_path = Path(tempfile.mkdtemp()) / "experiments.sqlite"
        store = ExperimentStore(db_path)
        store.ensure_schema()
        candidate = CandidateMetadata(
            hypothesis="h",
            rationale_summary="r",
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
        )
        descriptor = PatchDescriptor(
            touched_files=["src/train.py"],
            touched_symbols=["train_epoch"],
            patch_category="optimizer",
            operation_type="replace",
            numeric_knobs={"learning_rate": 0.001},
            rationale_summary="r",
            estimated_risk=0.2,
            parent_branch="boa/demo/accepted",
            parent_trial_id=None,
            budget_used="scout",
            diff_path="/tmp/patch.diff",
        )
        store.create_trial(
            run_tag="demo",
            trial_id="demo-0001",
            branch_name="boa/demo/trial/demo-0001",
            parent_branch="boa/demo/accepted",
            parent_trial_id=None,
            candidate=candidate,
            descriptor=descriptor,
            search_decision={"policy": "random"},
            diff_path=Path("/tmp/patch.diff"),
        )
        stage_result = StageRunResult(
            stage_name="scout",
            branch_name="boa/demo/trial/demo-0001",
            status="succeeded",
            command_results=[
                StageCommandResult(
                    index=1,
                    command="python train.py",
                    exit_code=0,
                    status="ok",
                    stdout_path=Path("/tmp/stdout"),
                    stderr_path=Path("/tmp/stderr"),
                    wall_time_seconds=1.2,
                )
            ],
            metrics={"accuracy": 0.9},
            primary_metric=0.9,
            cost_metric=None,
            adjusted_score=0.9,
            threshold_passed=True,
            improved=True,
            advanced=True,
            final_accept=False,
            reason="threshold_passed,improved,advance",
            started_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:02+00:00",
            resource_metadata={"hostname": "train-box"},
            artifact_dir=Path("/tmp/artifacts"),
        )
        store.record_stage_result(trial_id="demo-0001", result=stage_result)
        store.upsert_incumbent(
            stage_name="scout",
            trial_id="demo-0001",
            branch_name="boa/demo/trial/demo-0001",
            adjusted_score=0.9,
            primary_metric=0.9,
        )
        store.set_acceptance(
            trial_id="demo-0001",
            acceptance_status="staged_only",
            canonical_stage="scout",
            canonical_score=0.9,
        )
        recent = store.recent_trials(limit=5)
        incumbent = store.get_incumbent("scout")
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].trial_id, "demo-0001")
        self.assertEqual(recent[0].canonical_stage, "scout")
        self.assertIsNotNone(incumbent)
        self.assertEqual(incumbent.trial_id, "demo-0001")


if __name__ == "__main__":
    unittest.main()
