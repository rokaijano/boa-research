from __future__ import annotations

import unittest
from pathlib import Path

from boaresearch.schema import (
    BoaConfig,
    CandidateMetadata,
    MetricConfig,
    ObjectiveConfig,
    PatchDescriptor,
    RunnerConfig,
    RunnerStageConfig,
    SearchConfig,
    TrialSummary,
)
from boaresearch.search import RepoSearchState, build_search_policy


def build_trial(trial_id: str, score: float, category: str) -> TrialSummary:
    candidate = CandidateMetadata(
        hypothesis="h",
        rationale_summary=f"{category} update",
        patch_category=category,
        operation_type="replace",
        estimated_risk=0.2,
    )
    descriptor = PatchDescriptor(
        touched_files=["src/train.py"],
        touched_symbols=["train_epoch"],
        patch_category=category,
        operation_type="replace",
        numeric_knobs={"learning_rate": 0.001},
        rationale_summary="update",
        estimated_risk=0.2,
        parent_branch="boa/demo/accepted",
        parent_trial_id=None,
        budget_used="scout",
        diff_path="/tmp/patch.diff",
    )
    return TrialSummary(
        trial_id=trial_id,
        branch_name=f"boa/demo/trial/{trial_id}",
        parent_branch="boa/demo/accepted",
        parent_trial_id=None,
        acceptance_status="accepted",
        canonical_stage="promoted",
        canonical_score=score,
        candidate=candidate,
        descriptor=descriptor,
    )


def build_config(policy: str) -> BoaConfig:
    return BoaConfig(
        metrics=[MetricConfig(name="accuracy", source="regex", pattern="accuracy=([0-9.]+)")],
        objective=ObjectiveConfig(primary_metric="accuracy", direction="maximize"),
        search=SearchConfig(policy=policy, seed=7),
        runner=RunnerConfig(mode="local", scout=RunnerStageConfig(commands=["echo ok"])),
        boa_md_path=Path("boa.md"),
    )


class SearchTests(unittest.TestCase):
    def test_greedy_prefers_best_trial(self) -> None:
        config = build_config("greedy_best_first")
        policy = build_search_policy(config)
        decision = policy.propose(
            [build_trial("t1", 0.1, "data"), build_trial("t2", 0.4, "optimizer")],
            RepoSearchState(accepted_branch="boa/demo/accepted"),
            config,
        )
        self.assertEqual(decision.parent_trial_id, "t2")
        self.assertEqual(decision.patch_category_hint, "optimizer")

    def test_local_ranking_prefers_non_misc_family(self) -> None:
        config = build_config("local_ranking")
        policy = build_search_policy(config)
        decision = policy.propose(
            [build_trial("t1", 0.2, "misc"), build_trial("t2", 0.18, "optimizer")],
            RepoSearchState(accepted_branch="boa/demo/accepted"),
            config,
        )
        self.assertEqual(decision.parent_trial_id, "t2")


if __name__ == "__main__":
    unittest.main()
