from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from boaresearch.runtime import BoaController
from boaresearch.runtime.controller import ControllerStateError
from boaresearch.runtime.observer import RunEvent
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
    TrialReflection,
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

    def cleanup_scratch_artifacts(self) -> list[str]:
        return []

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
        stdout_path = artifact_dir / "stdout.txt"
        stderr_path = artifact_dir / "stderr.txt"
        stdout_path.write_text("epoch=1 train_loss=0.4 val_accuracy=0.72\n", encoding="utf-8")
        stderr_path.write_text("" if self.status == "succeeded" else f"{self.status}\n", encoding="utf-8")
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
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
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


class RetryThenPatchWorktree(FakeWorktree):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.validation_calls = 0

    def validate_changed_paths(self, *, allowed_paths, protected_paths):
        del allowed_paths, protected_paths
        self.validation_calls += 1
        if self.validation_calls == 1:
            return []
        return ["src/train.py"]


class FakeAgent:
    def __init__(
        self,
        *,
        plan_branch: str,
        bad_plan_call_id: bool = False,
        bad_candidate_call_id: bool = False,
        plan_lesson_ids: list[str] | None = None,
        candidate_lesson_ids: list[str] | None = None,
    ) -> None:
        self.plan_branch = plan_branch
        self.bad_plan_call_id = bad_plan_call_id
        self.bad_candidate_call_id = bad_candidate_call_id
        self.plan_lesson_ids = list(plan_lesson_ids or [])
        self.candidate_lesson_ids = None if candidate_lesson_ids is None else list(candidate_lesson_ids)
        self.prepare_called = False
        self.prepare_candidate_calls = 0
        self.reflect_trial_calls = 0
        self.plan_parent_trial_id: str | None = None

    def plan_trial(self, context) -> CandidatePlan:
        tool_context = read_search_tool_context(context.tool_context_path)
        recorder = SearchTraceRecorder(trace_path=tool_context.trace_path, phase=tool_context.phase)
        response = recorder.record(tool_name="list_lineage_options", request={"limit": 5}, response={"options": []})
        return CandidatePlan(
            hypothesis="h",
            rationale_summary="plan",
            selected_parent_branch=self.plan_branch,
            selected_parent_trial_id=(
                self.plan_parent_trial_id
                if self.plan_parent_trial_id is not None
                else (None if self.plan_branch == "boa/demo/accepted" else "prior-0001")
            ),
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            informed_by_call_ids=["missing-call"] if self.bad_plan_call_id else [response["call_id"]],
            addressed_lesson_ids=list(self.plan_lesson_ids),
        )

    def prepare_candidate(self, context) -> CandidateMetadata:
        self.prepare_called = True
        self.prepare_candidate_calls += 1
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
            addressed_lesson_ids=(
                list(self.plan_lesson_ids)
                if self.candidate_lesson_ids is None
                else list(self.candidate_lesson_ids)
            ),
        )

    def reflect_trial(self, context) -> TrialReflection:
        self.reflect_trial_calls += 1
        return TrialReflection(
            source_stage=context.source_stage,
            source_commands=[item.command for item in context.command_evidence],
            behavior_summary="Validation stalled after an early gain.",
            primary_problem="Generalization plateaued.",
            under_optimized=["regularization"],
            suggested_fixes=["Increase weight decay slightly."],
            evidence=["train loss fell while validation stalled"],
            outcome=f"Acceptance outcome: {context.acceptance_status}",
        )


class NoTraceAgent(FakeAgent):
    def plan_trial(self, context) -> CandidatePlan:
        del context
        return CandidatePlan(
            hypothesis="h",
            rationale_summary="plan",
            selected_parent_branch=self.plan_branch,
            selected_parent_trial_id=None,
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            informed_by_call_ids=["missing-call"],
            addressed_lesson_ids=[],
        )


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class ControllerTests(unittest.TestCase):
    def _controller(
        self,
        *,
        fake_agent: FakeAgent,
        repo: Path | None = None,
        runner: FakeRunner | None = None,
        worktree: FakeWorktree | None = None,
        observer: RecordingObserver | None = None,
    ) -> tuple[BoaController, Path]:
        repo = repo or Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "boa.md").write_text("# BOA\n", encoding="utf-8")
        (repo / "boa.config").write_text("schema_version = 3\n", encoding="utf-8")
        controller = BoaController(build_config(repo), observer=observer)
        controller.agent = fake_agent
        controller.runner = runner or FakeRunner()
        controller.worktree = worktree or FakeWorktree(repo / ".boa" / "worktree" / "demo")
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

    def _seed_lesson_trial(self, controller: BoaController) -> str:
        controller.store.ensure_schema()
        controller.store.create_trial(
            run_tag="demo",
            trial_id="seed-0001",
            branch_name="boa/demo/trial/seed-0001",
            parent_branch="boa/demo/accepted",
            parent_trial_id=None,
            candidate_plan=None,
            candidate=None,
            descriptor=None,
            search_trace=[],
            diff_path=None,
            acceptance_status="completed_rejected",
        )
        controller.store.update_trial_reflection(
            trial_id="seed-0001",
            reflection=TrialReflection(
                source_stage="scout",
                source_commands=["python train.py"],
                behavior_summary="Validation stalled after early gains.",
                primary_problem="Generalization plateaued.",
                under_optimized=["regularization"],
                suggested_fixes=["Increase weight decay slightly."],
                evidence=["train loss fell while validation stalled"],
                outcome="Patch was insufficient.",
            ),
        )
        controller.store.set_acceptance(
            trial_id="seed-0001",
            acceptance_status="completed_rejected",
            canonical_stage="scout",
            canonical_score=0.5,
        )
        return "seed-0001:lesson:1"

    def test_tool_context_trace_path_is_inside_protected_runtime(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/accepted"))
        controller._prepare_planning_workspace()

        context = controller._build_planning_context(trial_id="demo-0001", recent_trials=[])
        tool_context = read_search_tool_context(context.tool_context_path)

        self.assertIsNotNone(tool_context.trace_path)
        self.assertTrue(str(tool_context.trace_path).startswith(str(controller.paths.protected_root)))
        self.assertIn("/.boa/protected/agent_traces/demo-0001/search_calls.jsonl", str(tool_context.trace_path).replace("\\", "/"))
        self.assertTrue(str(context.plan_output_path).startswith(str(controller.paths.protected_root)))
        self.assertIn("/.boa/protected/agent_outputs/demo-0001/plan.json", str(context.plan_output_path).replace("\\", "/"))
        self.assertEqual(context.bo_suggestion_report["report_type"], "planning_bo_suggestions")
        self.assertEqual(context.trial_dataset, [])

    def test_execution_candidate_output_path_is_inside_protected_runtime(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/accepted"))
        controller._prepare_planning_workspace()

        candidate_plan = CandidatePlan(
            hypothesis="h",
            rationale_summary="plan",
            selected_parent_branch="boa/demo/accepted",
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            informed_by_call_ids=["boa-call-1"],
        )
        context = controller._build_execution_context(
            trial_id="demo-0001",
            trial_branch="boa/demo/trial/demo-0001",
            parent_branch="boa/demo/accepted",
            parent_trial_id=None,
            recent_trials=[],
            candidate_plan=candidate_plan,
        )

        self.assertTrue(str(context.candidate_output_path).startswith(str(controller.paths.protected_root)))
        self.assertIn("/.boa/protected/agent_outputs/demo-0001/candidate.json", str(context.candidate_output_path).replace("\\", "/"))

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
        self.assertIsNotNone(created.reflection)
        self.assertEqual(created.reflection.source_stage, "scout")

    def test_accepted_parent_with_nonempty_trial_id_is_normalized_and_runs(self) -> None:
        fake_agent = FakeAgent(plan_branch="boa/demo/accepted")
        fake_agent.plan_parent_trial_id = "prior-0001"
        controller, _repo = self._controller(fake_agent=fake_agent)

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        created = next(trial for trial in recent if trial.trial_id == "demo-0001")
        self.assertEqual(created.acceptance_status, "completed_accepted")

    def test_invalid_parent_is_rejected_before_execution(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/trial/unknown"))

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "policy_rejected")
        self.assertFalse(controller.agent.prepare_called)
        self.assertEqual(controller.agent.reflect_trial_calls, 0)

    def test_unknown_call_ids_are_rejected(self) -> None:
        controller, _repo = self._controller(fake_agent=FakeAgent(plan_branch="boa/demo/accepted", bad_candidate_call_id=True))

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "policy_rejected")

    def test_unknown_lesson_ids_are_rejected(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted", plan_lesson_ids=["missing:lesson:1"])
        )

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "policy_rejected")

    def test_candidate_cannot_expand_lesson_ids_beyond_plan(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(
                plan_branch="boa/demo/accepted",
                plan_lesson_ids=["seed-0001:lesson:1"],
                candidate_lesson_ids=["seed-0001:lesson:1", "seed-0001:lesson:2"],
            )
        )
        self._seed_lesson_trial(controller)

        controller.run()

        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "policy_rejected")

    def test_missing_trace_gives_clear_policy_rejection_detail(self) -> None:
        controller, _repo = self._controller(fake_agent=NoTraceAgent(plan_branch="boa/demo/accepted"))

        summary = controller.run()

        self.assertEqual(summary.last_acceptance_status, "policy_rejected")
        self.assertIn("Unknown BOA tool call ids referenced", summary.last_detail or "")

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
            worktree=NoPatchWorktree(repo / ".boa" / "worktree" / "demo"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "no_patch")
        self.assertIsNone(trial.reflection)
        self.assertEqual(controller.agent.reflect_trial_calls, 0)

    def test_no_patch_retries_once_before_failing(self) -> None:
        repo = Path(tempfile.mkdtemp())
        fake_agent = FakeAgent(plan_branch="boa/demo/accepted")
        controller, _repo = self._controller(
            fake_agent=fake_agent,
            repo=repo,
            worktree=NoPatchWorktree(repo / ".boa" / "worktree" / "demo"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "no_patch")
        self.assertEqual(fake_agent.prepare_candidate_calls, 2)

    def test_no_patch_retry_can_recover_and_complete_trial(self) -> None:
        repo = Path(tempfile.mkdtemp())
        fake_agent = FakeAgent(plan_branch="boa/demo/accepted")
        controller, _repo = self._controller(
            fake_agent=fake_agent,
            repo=repo,
            worktree=RetryThenPatchWorktree(repo / ".boa" / "worktree" / "demo"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "completed_accepted")
        self.assertEqual(fake_agent.prepare_candidate_calls, 2)

    def test_timeout_is_recorded_distinctly(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            runner=FakeRunner(status="timeout"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "timed_out")
        self.assertIsNotNone(trial.reflection)
        self.assertEqual(trial.reflection.source_stage, "scout")

    def test_metric_missing_is_recorded_distinctly(self) -> None:
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            runner=FakeRunner(status="metric_missing"),
        )
        controller.run()
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        trial = next(item for item in recent if item.trial_id.endswith("0001"))
        self.assertEqual(trial.acceptance_status, "metric_missing")
        self.assertIsNotNone(trial.reflection)
        self.assertEqual(trial.reflection.source_stage, "scout")

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

    def test_ensure_ready_rejects_repo_without_initial_commit(self) -> None:
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "boa.md").write_text("# BOA\n", encoding="utf-8")
        (repo / "boa.config").write_text("schema_version = 3\n", encoding="utf-8")

        controller = BoaController(build_config(repo))

        with self.assertRaises(ControllerStateError) as exc:
            controller._ensure_ready()  # noqa: SLF001

        self.assertIn("requires at least one commit", str(exc.exception))
        self.assertIn("main", str(exc.exception))

    def test_seed_baseline_metric_missing_raises_clear_error_and_records_status(self) -> None:
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "boa.md").write_text("# BOA\n", encoding="utf-8")
        (repo / "boa.config").write_text("schema_version = 3\n", encoding="utf-8")

        controller = BoaController(build_config(repo))
        controller.agent = FakeAgent(plan_branch="boa/demo/accepted")
        controller.runner = FakeRunner(status="metric_missing")
        controller.worktree = FakeWorktree(repo / ".boa" / "worktree" / "demo")
        controller.store.ensure_schema()
        controller._ensure_ready = lambda: controller.store.ensure_schema()  # type: ignore[method-assign]
        controller._run_preflight = lambda: None  # type: ignore[method-assign]

        with self.assertRaises(ControllerStateError) as exc:
            controller._seed_baseline("scout")  # noqa: SLF001

        self.assertIn("Accepted-branch baseline failed", str(exc.exception))
        recent = controller.store.recent_trials(limit=5, run_tag="demo")
        baseline = next(item for item in recent if item.trial_id == "demo-baseline-scout")
        self.assertEqual(baseline.acceptance_status, "metric_missing")

    def test_summarize_exception_detail_prefers_json_message(self) -> None:
        exc = RuntimeError(
            'CLI agent failed.\nstderr:\nERROR: {"error":{"message":"Invalid schema: Missing \\"notes\\"."},"status":400}'
        )
        self.assertEqual(
            BoaController._summarize_exception_detail(exc),  # noqa: SLF001
            'Invalid schema: Missing "notes".',
        )

    def test_controller_emits_progress_events(self) -> None:
        observer = RecordingObserver()
        controller, _repo = self._controller(
            fake_agent=FakeAgent(plan_branch="boa/demo/accepted"),
            observer=observer,
        )

        controller.run()

        kinds = [event.kind for event in observer.events]
        self.assertIn("run_started", kinds)
        self.assertIn("trial_started", kinds)
        self.assertIn("planning_started", kinds)
        self.assertIn("execution_started", kinds)
        self.assertIn("stage_started", kinds)
        self.assertIn("trial_completed", kinds)


if __name__ == "__main__":
    unittest.main()
