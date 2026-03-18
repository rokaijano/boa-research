from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boaresearch.schema import (
    BoaConfig,
    CandidateMetadata,
    MetricConfig,
    ObjectiveConfig,
    PatchDescriptor,
    RunConfig,
    RunnerConfig,
    RunnerStageConfig,
    SearchConfig,
    TrialSummary,
)
from boaresearch.search import SearchOracleService, SearchToolbox, SearchTraceRecorder


def build_trial(trial_id: str, score: float, category: str, *, knobs: dict[str, float] | None = None) -> TrialSummary:
    candidate = CandidateMetadata(
        hypothesis="h",
        rationale_summary=f"{category} update",
        patch_category=category,
        operation_type="replace",
        estimated_risk=0.2,
        informed_by_call_ids=["boa-call-seed"],
    )
    descriptor = PatchDescriptor(
        touched_files=["src/train.py"],
        touched_symbols=["train_epoch"],
        patch_category=category,
        operation_type="replace",
        numeric_knobs=dict(knobs or {"learning_rate": 0.001}),
        rationale_summary="update",
        estimated_risk=0.2,
        parent_branch="boa/demo/accepted",
        parent_trial_id=None,
        budget_used="scout",
        diff_path="/tmp/patch.diff",
    )
    return TrialSummary(
        trial_id=trial_id,
        run_tag="demo",
        branch_name=f"boa/demo/trial/{trial_id}",
        parent_branch="boa/demo/accepted",
        parent_trial_id=None,
        acceptance_status="accepted",
        canonical_stage="promoted",
        canonical_score=score,
        candidate_plan=None,
        candidate=candidate,
        descriptor=descriptor,
    )


def build_config() -> BoaConfig:
    return BoaConfig(
        run=RunConfig(tag="demo", accepted_branch="boa/demo/accepted"),
        metrics=[MetricConfig(name="accuracy", source="regex", pattern="accuracy=([0-9.]+)")],
        objective=ObjectiveConfig(primary_metric="accuracy", direction="maximize"),
        search=SearchConfig(oracle="bayesian_optimization", seed=7),
        runner=RunnerConfig(mode="local", scout=RunnerStageConfig(commands=["echo ok"])),
        boa_md_path=Path("boa.md"),
    )


class SearchTests(unittest.TestCase):
    def test_list_lineage_options_returns_accepted_plus_trials(self) -> None:
        oracle = SearchOracleService(
            config=build_config(),
            memory=[build_trial("t1", 0.2, "optimizer")],
            accepted_branch="boa/demo/accepted",
        )
        options = oracle.list_lineage_options()
        self.assertEqual(options[0]["branch_name"], "boa/demo/accepted")
        self.assertEqual(options[1]["trial_id"], "t1")

    def test_suggest_parents_prefers_request_aligned_parent(self) -> None:
        oracle = SearchOracleService(
            config=build_config(),
            memory=[build_trial("t1", 0.72, "misc"), build_trial("t2", 0.68, "optimizer")],
            accepted_branch="boa/demo/accepted",
        )
        suggestions = oracle.suggest_parents({"patch_category": "optimizer", "operation_type": "replace", "estimated_risk": 0.2})
        self.assertEqual(suggestions[0]["trial_id"], "t2")

    def test_rank_patch_families_prefers_optimizer(self) -> None:
        oracle = SearchOracleService(
            config=build_config(),
            memory=[build_trial("t1", 0.2, "misc"), build_trial("t2", 0.7, "optimizer")],
            accepted_branch="boa/demo/accepted",
        )
        families = oracle.rank_patch_families({"operation_type": "replace", "estimated_risk": 0.2})
        self.assertEqual(families[0]["patch_category"], "optimizer")

    def test_score_candidate_descriptor_reports_acquisition_fields(self) -> None:
        oracle = SearchOracleService(
            config=build_config(),
            memory=[build_trial("t1", 0.3, "data"), build_trial("t2", 0.7, "optimizer")],
            accepted_branch="boa/demo/accepted",
        )
        score = oracle.score_candidate_descriptor(
            {
                "patch_category": "optimizer",
                "operation_type": "replace",
                "estimated_risk": 0.25,
                "numeric_knobs": {"learning_rate": 0.0003},
            }
        )
        self.assertIn("acquisition_score", score)
        self.assertIn("posterior_mean", score)
        self.assertIn("posterior_std", score)

    def test_propose_numeric_knob_regions_aggregates_history(self) -> None:
        oracle = SearchOracleService(
            config=build_config(),
            memory=[
                build_trial("t1", 0.5, "optimizer", knobs={"learning_rate": 0.001, "weight_decay": 0.01}),
                build_trial("t2", 0.7, "optimizer", knobs={"learning_rate": 0.0003, "weight_decay": 0.02}),
            ],
            accepted_branch="boa/demo/accepted",
        )
        regions = oracle.propose_numeric_knob_regions({"patch_category": "optimizer"})
        self.assertEqual(regions[0]["name"], "learning_rate")
        self.assertGreater(regions[0]["count"], 1)

    def test_toolbox_emits_call_ids(self) -> None:
        trace_path = Path(tempfile.mkdtemp()) / "search_calls.jsonl"
        toolbox = SearchToolbox(
            oracle=SearchOracleService(
                config=build_config(),
                memory=[build_trial("t1", 0.5, "optimizer")],
                accepted_branch="boa/demo/accepted",
            ),
            recorder=SearchTraceRecorder(trace_path=trace_path, phase="planning"),
        )
        response = toolbox.list_lineage_options({"limit": 3})
        self.assertIn("call_id", response)
        self.assertTrue(trace_path.exists())


if __name__ == "__main__":
    unittest.main()
