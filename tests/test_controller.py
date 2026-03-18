from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from boaresearch.runtime import BoaController
from boaresearch.runner import StageExecution
from boaresearch.schema import (
    BoaConfig,
    CandidateMetadata,
    CandidatePlan,
    MetricConfig,
    ObjectiveConfig,
    RunConfig,
    RunnerConfig,
    RunnerStageConfig,
    SearchConfig,
    StageCommandResult,
)
from boaresearch.search import SearchTraceRecorder, read_search_tool_context


def build_config(repo: Path) -> BoaConfig:
    return BoaConfig(
        repo_root=repo,
        boa_md_path=repo / "boa.md",
        config_path=repo / "boa.config",
        run=RunConfig(tag="demo", max_trials=1, max_consecutive_failures=3, accepted_branch="boa/demo/accepted"),
        metrics=[MetricConfig(name="accuracy", source="regex", pattern="accuracy=([0-9.]+)")],
        objective=ObjectiveConfig(primary_metric="accuracy", direction="maximize"),
        search=SearchConfig(oracle="bayesian_optimization"),
        runner=RunnerConfig(mode="local", scout=RunnerStageConfig(commands=["echo ok"])),
    )


class FakeWorktree:
    def __init__(self, path: Path) -> None:
        self.worktree_path = path
        self.prepared: list[tuple[str, str]] = []
        self.promoted: list[str] = []

    def prepare_trial(self, *, trial_branch: str, parent_branch: str) -> None:
        self.prepared.append((trial_branch, parent_branch))
        self.worktree_path.mkdir(parents=True, exist_ok=True)

    def validate_changed_paths(self, *, allowed_paths, protected_paths):
        return ["src/train.py"]

    def diff_text(self, *, base_ref: str) -> str:
        del base_ref
        return (
            "diff --git a/src/train.py b/src/train.py\n"
            "--- a/src/train.py\n"
            "+++ b/src/train.py\n"
            "@@ -1 +1,2 @@ train_epoch\n"
            "+learning_rate = 0.0003\n"
        )

    def commit_trial(self, message: str) -> str:
        return message

    def promote_trial(self, *, trial_branch: str) -> str:
        self.promoted.append(trial_branch)
        return trial_branch


class FakeRunner:
    def __init__(self, *, status: str = "succeeded", metrics: dict[str, float] | None = None) -> None:
        self.status = status
        self.metrics = dict(metrics or {"accuracy": 0.9})

    requires_remote_branches = False

    def run_stage(self, *, trial_id, branch_name, worktree_path, stage_name, stage, metrics, artifact_dir):
        del trial_id, worktree_path, stage, metrics
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return StageExecution(
            stage_name=stage_name,
            branch_name=branch_name,
            status=self.status,
            command_results=[
                StageCommandResult(
                    index=1,
                    command="echo ok",
                    exit_code=0,
                    status="ok",
                    stdout_path=artifact_dir / "stdout.txt",
                    stderr_path=artifact_dir / "stderr.txt",
                    wall_time_seconds=0.1,
                )
            ],
            metrics=dict(self.metrics if self.status == "succeeded" else {}),
            resource_metadata={"hostname": "local"},
            artifact_dir=artifact_dir,
            started_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T00:00:01+00:00",
        )


class NoPatchWorktree(FakeWorktree):
    def validate_changed_paths(self, *, allowed_paths, protected_paths):
        del allowed_paths, protected_paths
        return []


class FakeAgent:
    def __init__(self, *, plan_branch: str, bad_plan_call_id: bool = False, bad_candidate_call_id: bool = False) -> None:
        self.plan_branch = plan_branch
        self.bad_plan_call_id = bad_plan_call_id
        self.bad_candidate_call_id = bad_candidate_call_id
        self.prepare_called = False

    def plan_trial(self, context) -> CandidatePlan:
        tool_context = read_search_tool_context(context.tool_context_path)
        recorder = SearchTraceRecorder(trace_path=tool_context.trace_path, phase=tool_context.phase)
        response = recorder.record(tool_name="list_lineage_options", request={"limit": 5}, response={"options": []})
        return CandidatePlan(
            hypothesis="h",
            rationale_summary="plan",
            selected_parent_branch=self.plan_branch,
            selected_parent_trial_id=None if self.plan_branch == "boa/demo/accepted" else "prior-0001",
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            informed_by_call_ids=["missing-call"] if self.bad_plan_call_id else [response["call_id"]],
        )

    def prepare_candidate(self, context) -> CandidateMetadata:
        self.prepare_called = True
        tool_context = read_search_tool_context(context.tool_context_path)
        recorder = SearchTraceRecorder(trace_path=tool_context.trace_path, phase=tool_context.phase)
        response = recorder.record(
            tool_name="score_candidate_descriptor",
            request={"patch_category": "optimizer"},
            response={"acquisition_score": 1.0},
        )
        return CandidateMetadata(
            hypothesis="h",
            rationale_summary="candidate",
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            numeric_knobs={"learning_rate": 0.0003},
            target_symbols=["train_epoch"],
            informed_by_call_ids=["missing-call"] if self.bad_candidate_call_id else [response["call_id"]],
        )


class ControllerTests(unittest.TestCase):
    def _controller(
        self,
        *,
        fake_agent: FakeAgent,
        repo: Path | None = None,
        runner: FakeRunner | None = None,
        worktree: FakeWorktree | None = None,
    ) -> tuple[BoaController, Path]:
        repo = repo or Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "boa.md").write_text("# BOA\n", encoding="utf-8")
        (repo / "boa.config").write_text("schema_version = 3\n", encoding="utf-8")
        controller = BoaController(build_config(repo))
        controller.agent = fake_agent
        controller.runner = runner or FakeRunner()
        controller.worktree = worktree or FakeWorktree(repo / ".boa" / "worktrees" / "demo")
        controller._ensure_ready = lambda: controller.store.ensure_schema()  # type: ignore[method-assign]
        controller._seed_baseline = lambda stage_name: None  # type: ignore[method-assign]
        controller._run_preflight = lambda: None  # type: ignore[method-assign]
        return controller, repo

    def _seed_prior_trial(self, controller: BoaController) -> None:
        controller.store.ensure_schema()
        controller.store.create_trial(
            run_tag="demo",
            trial_id="prior-0001",
            branch_name="boa/demo/trial/prior-0001",
            parent_branch="boa/demo/accepted",
            parent_trial_id=None,
            candidate_plan=None,
            candidate=None,
            descriptor=None,
            search_trace=[],
            diff_path=None,
            acceptance_status="accepted",
        )
        controller.store.set_acceptance(
            trial_id="prior-0001",
            acceptance_status="accepted",
            canonical_stage="scout",
            canonical_score=0.7,
        )

    def test_run_uses_prior_trial_parent_and_accepts(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/trial/prior-0001"))
        self._seed_prior_trial(controller)

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        created = next(trial for trial in recent if trial.trial_id == "demo-0001")
        self.assertEqual(controller.worktree.prepared[0], ("boa/demo/accepted", "boa/demo/accepted"))
        self.assertEqual(controller.worktree.prepared[1][1], "boa/demo/trial/prior-0001")
        self.assertTrue(controller.worktree.promoted)
        self.assertEqual(created.acceptance_status, "completed_accepted")

    def test_invalid_parent_is_rejected_before_execution(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/trial/unknown"))

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "policy_rejected")
        self.assertFalse(controller.agent.prepare_called)

    def test_unknown_call_ids_are_rejected(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/accepted", bad_candidate_call_id=True))

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "policy_rejected")

    def test_second_controller_run_uses_next_trial_id(self) -> None:
        repo = Path(tempfile.mkdtemp())
        controller1, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/accepted"), repo=repo)
        controller1.run()
        controller2, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/accepted"), repo=repo)
        controller2.run()
        recent = controller2.store.recent_trials(limit=10, run_tag="demo")
        self.assertEqual(sorted(trial.trial_id for trial in recent), ["demo-0001", "demo-0002"])

    def test_no_patch_is_recorded_distinctly(self) -> None:
        repo = Path(tempfile.mkdtemp())
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            repo=repo,
            worktree=NoPatchWorktree(repo / ".boa" / "worktrees" / "demo"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "no_patch")

    def test_timeout_is_recorded_distinctly(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            runner=FakeRunner(status="timeout"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "timed_out")

    def test_metric_missing_is_recorded_distinctly(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            runner=FakeRunner(status="metric_missing"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "metric_missing")

    def test_non_improving_trial_is_completed_rejected(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            runner=FakeRunner(metrics={"accuracy": 0.4}),
        )
        controller.store.ensure_schema()
        controller.store.upsert_incumbent(
            stage_name="scout",
            trial_id="seed-0001",
            branch_name="boa/demo/trial/seed-0001",
            adjusted_score=0.8,
            primary_metric=0.8,
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "completed_rejected")
        self.assertEqual(trial.canonical_stage, "scout")


if __name__ == "__main__":
    unittest.main()
