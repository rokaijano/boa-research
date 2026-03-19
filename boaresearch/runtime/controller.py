from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import git_state
from ..acceptance import AcceptanceEngine
from ..agents import ResearchAgentError, build_agent
from ..descriptors import build_patch_descriptor
from ..git_auth import GitAuthManager
from ..loader import enabled_stages
from .paths import BoaPaths
from .observer import NullRunObserver, RunEvent
from ..runner import RunnerError, build_trial_runner
from ..schema import (
    AgentExecutionContext,
    AgentPlanningContext,
    BoaConfig,
    CandidateMetadata,
    CandidatePlan,
    SearchToolCall,
    StageRunResult,
    TrialSummary,
)
from ..search import SearchOracleService, SearchToolContext, SearchToolbox, SearchTraceRecorder, load_search_trace, write_search_tool_context
from .store import ExperimentStore
from .worktree import WorktreeError, WorktreeManager


class PolicyRejectedError(RuntimeError):
    pass


class ControllerStateError(RuntimeError):
    pass


@dataclass
class RunSummary:
    trials_attempted: int
    stop_requested: bool
    last_trial_id: str | None
    last_acceptance_status: str | None
    last_canonical_stage: str | None
    last_canonical_score: float | None
    last_detail: str | None


class BoaController:
    def __init__(self, config: BoaConfig, *, observer=None) -> None:
        self.config = config
        self.observer = observer or NullRunObserver()
        self.paths = BoaPaths.from_config(config)
        helper_root = self.paths.protected_root / "git_auth_helpers"
        self.git_auth = GitAuthManager(config.git_auth, helper_root=helper_root)
        self.store = ExperimentStore(self.paths.store_path)
        self.acceptance = AcceptanceEngine(config)
        self.runner = build_trial_runner(config, git_auth=self.git_auth, observer=self.observer)
        self.worktree = WorktreeManager(
            repo_root=config.repo_root,
            worktree_path=config.run.worktree_path or self.paths.worktree_dir(config.run.tag),
            accepted_branch=str(config.run.accepted_branch),
            git_auth=self.git_auth,
        )
        self.agent = build_agent(
            config,
            repo_root=config.repo_root,
            run_preflight=self._run_preflight,
            observer=self.observer,
        )

    def _emit(
        self,
        *,
        kind: str,
        message: str,
        trial_id: str | None = None,
        phase: str | None = None,
        stage_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.observer.emit(
            RunEvent(
                kind=kind,
                message=message,
                trial_id=trial_id,
                phase=phase,
                stage_name=stage_name,
                metadata=dict(metadata or {}),
            )
        )

    def _objective_summary(self) -> str:
        parts = [
            f"Objective: {self.config.objective.direction} `{self.config.objective.primary_metric}`.",
            f"Minimum improvement delta: {self.config.objective.minimum_improvement_delta}.",
        ]
        if self.config.objective.threshold is not None:
            parts.append(f"Threshold: {self.config.objective.threshold}.")
        if self.config.objective.cost_penalty_metric:
            parts.append(
                f"Cost penalty: {self.config.objective.cost_penalty_weight} * `{self.config.objective.cost_penalty_metric}`."
            )
        return " ".join(parts)

    def _run_command(self, *, command: str, cwd: Path) -> None:
        args = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]
        proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Preflight command failed: {command}\n{detail}")

    def _run_preflight(self) -> None:
        for command in self.config.guardrails.preflight_commands:
            self._emit(
                kind="preflight_started",
                message="Running preflight checks",
            )
            self._run_command(command=command, cwd=self.worktree.worktree_path)
            self._emit(
                kind="preflight_completed",
                message="Preflight checks completed",
            )

    def _ensure_ready(self) -> None:
        self.paths.ensure()
        self.store.ensure_schema()
        if self.config.run.base_branch is None:
            self.config.run.base_branch = git_state.current_branch(self.config.repo_root)
        if not git_state.has_commits(self.config.repo_root):
            branch = str(self.config.run.base_branch or "HEAD")
            raise ControllerStateError(
                f"BOA run requires at least one commit in the target repository. "
                f"The current branch is '{branch}', but HEAD does not point to a commit yet. "
                "Create an initial commit and rerun `boa run`."
            )
        self.worktree.ensure_accepted_branch(base_branch=str(self.config.run.base_branch))
        if self.runner.requires_remote_branches:
            self.worktree.push_branch(
                remote=self.config.runner.ssh.git_remote,
                branch=str(self.config.run.accepted_branch),
                force=False,
            )

    def _next_trial_id(self) -> str:
        sequence = self.store.next_trial_sequence(run_tag=self.config.run.tag)
        return f"{self.config.run.tag}-{sequence:04d}"

    def _trial_branch(self, trial_id: str) -> str:
        return f"boa/{self.config.run.tag}/trial/{trial_id}"

    def _trial_artifact_dir(self, trial_id: str) -> Path:
        path = self.paths.trial_artifact_dir(trial_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _search_trace_path(self, trial_id: str) -> Path:
        path = self.paths.search_trace_path(trial_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _agent_output_dir(self, trial_id: str) -> Path:
        path = self.paths.agent_output_dir(trial_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _plan_output_path(self, trial_id: str) -> Path:
        return self._agent_output_dir(trial_id) / "plan.json"

    def _candidate_output_path(self, trial_id: str) -> Path:
        return self._agent_output_dir(trial_id) / "candidate.json"

    def _artifact_plan_path(self, trial_id: str) -> Path:
        return self._trial_artifact_dir(trial_id) / "plan.json"

    def _artifact_candidate_path(self, trial_id: str) -> Path:
        return self._trial_artifact_dir(trial_id) / "candidate.json"

    def _prompt_bundle_dir(self, trial_id: str, *, phase: str) -> Path:
        path = self.paths.prompt_bundle_dir(trial_id) / phase
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _tool_context_path(self, trial_id: str, *, phase: str) -> Path:
        return self._prompt_bundle_dir(trial_id, phase=phase) / "tool_context.json"

    def _write_tool_context(self, *, trial_id: str, phase: str) -> Path:
        path = self._tool_context_path(trial_id, phase=phase)
        write_search_tool_context(
            path,
            SearchToolContext(
                repo_root=self.config.repo_root,
                config_path=self.config.config_path,
                run_tag=self.config.run.tag,
                trial_id=trial_id,
                accepted_branch=str(self.config.run.accepted_branch),
                phase=phase,
                trace_path=self._search_trace_path(trial_id),
            ),
        )
        return path

    def _commit_message(self, trial_id: str, candidate: CandidateMetadata) -> str:
        return f"boa: {candidate.patch_category} trial {trial_id}"

    def _recent_trials(self) -> list[TrialSummary]:
        return self.store.recent_trials(limit=self.config.search.max_history, run_tag=self.config.run.tag)

    def _search_oracle(self, recent_trials: list[TrialSummary]) -> SearchOracleService:
        return SearchOracleService(
            config=self.config,
            memory=recent_trials,
            accepted_branch=str(self.config.run.accepted_branch),
        )

    def _prepare_planning_workspace(self) -> None:
        accepted_branch = str(self.config.run.accepted_branch)
        self.worktree.prepare_trial(trial_branch=accepted_branch, parent_branch=accepted_branch)

    def _build_planning_context(self, *, trial_id: str, recent_trials: list[TrialSummary]) -> AgentPlanningContext:
        bootstrap_tool_calls = self._record_planning_bootstrap_calls(trial_id=trial_id, recent_trials=recent_trials)
        return AgentPlanningContext(
            repo_root=self.config.repo_root,
            worktree_path=self.worktree.worktree_path,
            trial_id=trial_id,
            run_tag=self.config.run.tag,
            accepted_branch=str(self.config.run.accepted_branch),
            boa_md_path=self.config.boa_md_path,
            extra_context_files=list(self.config.agent.extra_context_files),
            allowed_paths=list(self.config.guardrails.allowed_paths),
            protected_paths=list(self.config.guardrails.protected_paths),
            recent_trials=recent_trials,
            bootstrap_tool_calls=bootstrap_tool_calls,
            objective_summary=self._objective_summary(),
            max_agent_steps=int(self.config.agent.max_agent_steps),
            prompt_bundle_dir=self._prompt_bundle_dir(trial_id, phase="planning"),
            tool_context_path=self._write_tool_context(trial_id=trial_id, phase="planning"),
            plan_output_path=self._plan_output_path(trial_id),
        )

    def _build_execution_context(
        self,
        *,
        trial_id: str,
        trial_branch: str,
        parent_branch: str,
        parent_trial_id: str | None,
        recent_trials: list[TrialSummary],
        candidate_plan: CandidatePlan,
    ) -> AgentExecutionContext:
        return AgentExecutionContext(
            repo_root=self.config.repo_root,
            worktree_path=self.worktree.worktree_path,
            trial_id=trial_id,
            run_tag=self.config.run.tag,
            accepted_branch=str(self.config.run.accepted_branch),
            trial_branch=trial_branch,
            parent_branch=parent_branch,
            parent_trial_id=parent_trial_id,
            boa_md_path=self.config.boa_md_path,
            extra_context_files=list(self.config.agent.extra_context_files),
            allowed_paths=list(self.config.guardrails.allowed_paths),
            protected_paths=list(self.config.guardrails.protected_paths),
            recent_trials=recent_trials,
            bootstrap_tool_calls=load_search_trace(self._search_trace_path(trial_id)),
            objective_summary=self._objective_summary(),
            preflight_commands=list(self.config.guardrails.preflight_commands),
            max_agent_steps=int(self.config.agent.max_agent_steps),
            prompt_bundle_dir=self._prompt_bundle_dir(trial_id, phase="execution"),
            tool_context_path=self._write_tool_context(trial_id=trial_id, phase="execution"),
            plan_output_path=self._plan_output_path(trial_id),
            candidate_output_path=self._candidate_output_path(trial_id),
            candidate_plan=candidate_plan,
        )

    def _record_planning_bootstrap_calls(self, *, trial_id: str, recent_trials: list[TrialSummary]) -> list[SearchToolCall]:
        trace_path = self._search_trace_path(trial_id)
        oracle = self._search_oracle(recent_trials)
        toolbox = SearchToolbox(
            oracle=oracle,
            recorder=SearchTraceRecorder(trace_path=trace_path, phase="planning", observer=self.observer),
        )
        toolbox.recent_trials({"limit": self.config.search.max_history})
        toolbox.list_lineage_options({"limit": self.config.search.max_history})
        return load_search_trace(trace_path)

    def _stage_config(self, stage_name: str):
        return getattr(self.config.runner, stage_name)

    def _baseline_failure_status(self, execution_status: str) -> str:
        if execution_status == "timeout":
            return "timed_out"
        if execution_status == "metric_missing":
            return "metric_missing"
        return "stage_failed"

    def _execution_failure_detail(self, execution) -> str:
        for command_result in execution.command_results:
            if command_result.status == "ok":
                continue
            detail = f" Failing command: {command_result.command}."
            stderr_text = ""
            if command_result.stderr_path.exists():
                stderr_text = command_result.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            elif command_result.stdout_path.exists():
                stderr_text = command_result.stdout_path.read_text(encoding="utf-8", errors="replace").strip()
            if stderr_text:
                last_line = stderr_text.splitlines()[-1].strip()
                if last_line:
                    detail += f" Detail: {last_line}"
            return detail
        return ""

    def _record_baseline_stage_result(
        self,
        *,
        trial_id: str,
        stage_name: str,
        execution,
        acceptance_status: str,
        reason: str,
    ) -> None:
        stage_result = StageRunResult(
            stage_name=stage_name,
            branch_name=str(self.config.run.accepted_branch),
            status=execution.status,
            command_results=execution.command_results,
            metrics=execution.metrics,
            primary_metric=None,
            cost_metric=None,
            adjusted_score=None,
            threshold_passed=False,
            improved=False,
            advanced=False,
            final_accept=False,
            reason=reason,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            resource_metadata=execution.resource_metadata,
            artifact_dir=execution.artifact_dir,
        )
        self.store.record_stage_result(trial_id=trial_id, result=stage_result)
        self.store.set_acceptance(
            trial_id=trial_id,
            acceptance_status=acceptance_status,
            canonical_stage=None,
            canonical_score=None,
        )

    def _seed_baseline(self, stage_name: str) -> None:
        if self.store.get_incumbent(stage_name) is not None:
            return
        self._emit(
            kind="baseline_started",
            message=f"Seeding accepted-branch baseline for stage '{stage_name}'",
            stage_name=stage_name,
        )
        trial_id = f"{self.config.run.tag}-baseline-{stage_name}"
        artifact_dir = self._trial_artifact_dir(trial_id)
        if not self.runner.requires_remote_branches:
            self.worktree.prepare_trial(
                trial_branch=str(self.config.run.accepted_branch),
                parent_branch=str(self.config.run.accepted_branch),
            )
        if not self.store.has_trial(trial_id):
            self.store.create_trial(
                run_tag=self.config.run.tag,
                trial_id=trial_id,
                branch_name=str(self.config.run.accepted_branch),
                parent_branch=str(self.config.run.accepted_branch),
                parent_trial_id=None,
                candidate_plan=None,
                candidate=None,
                descriptor=None,
                search_trace=[],
                diff_path=None,
                acceptance_status="baseline",
            )
        execution = self.runner.run_stage(
            trial_id=trial_id,
            branch_name=str(self.config.run.accepted_branch),
            worktree_path=self.worktree.worktree_path,
            stage_name=stage_name,
            stage=self._stage_config(stage_name),
            metrics=self.config.metrics,
            artifact_dir=artifact_dir,
        )
        if execution.status != "succeeded":
            acceptance_status = self._baseline_failure_status(execution.status)
            self._record_baseline_stage_result(
                trial_id=trial_id,
                stage_name=stage_name,
                execution=execution,
                acceptance_status=acceptance_status,
                reason=execution.status,
            )
            raise ControllerStateError(
                f"Accepted-branch baseline failed at stage '{stage_name}' with status '{acceptance_status}'. "
                "BOA cannot continue until the configured stage commands and metric extraction work on the current accepted branch."
                f"{self._execution_failure_detail(execution)}"
            )
        try:
            evaluation = self.acceptance.evaluate_stage(stage_name=stage_name, metrics=execution.metrics, incumbent=None)
        except ValueError as exc:
            self._record_baseline_stage_result(
                trial_id=trial_id,
                stage_name=stage_name,
                execution=execution,
                acceptance_status="metric_missing",
                reason=str(exc),
            )
            raise ControllerStateError(
                f"Accepted-branch baseline is missing required metrics at stage '{stage_name}': {exc}. "
                "BOA cannot continue until the eval command and metric extraction are aligned with boa.config."
            ) from exc
        stage_result = StageRunResult(
            stage_name=stage_name,
            branch_name=str(self.config.run.accepted_branch),
            status=execution.status,
            command_results=execution.command_results,
            metrics=execution.metrics,
            primary_metric=evaluation.primary_metric,
            cost_metric=evaluation.cost_metric,
            adjusted_score=evaluation.adjusted_score,
            threshold_passed=evaluation.threshold_passed,
            improved=evaluation.improved,
            advanced=evaluation.advanced,
            final_accept=False,
            reason=f"baseline:{evaluation.reason}",
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            resource_metadata=execution.resource_metadata,
            artifact_dir=execution.artifact_dir,
        )
        self.store.record_stage_result(trial_id=trial_id, result=stage_result)
        self.store.upsert_incumbent(
            stage_name=stage_name,
            trial_id=trial_id,
            branch_name=str(self.config.run.accepted_branch),
            adjusted_score=evaluation.adjusted_score,
            primary_metric=evaluation.primary_metric,
        )
        self.store.set_acceptance(
            trial_id=trial_id,
            acceptance_status="baseline",
            canonical_stage=stage_name,
            canonical_score=evaluation.adjusted_score,
        )
        self._emit(
            kind="baseline_completed",
            message=f"Baseline ready for stage '{stage_name}'",
            trial_id=trial_id,
            stage_name=stage_name,
            metadata={
                "score": evaluation.adjusted_score,
                "primary_metric": evaluation.primary_metric,
                "branch_name": str(self.config.run.accepted_branch),
                "parent_branch": str(self.config.run.accepted_branch),
                "parent_trial_id": None,
                "acceptance_status": "baseline",
                "canonical_stage": stage_name,
                "canonical_score": evaluation.adjusted_score,
            },
        )

    def _write_plan_artifact(self, *, trial_id: str, candidate_plan: CandidatePlan) -> None:
        self._artifact_plan_path(trial_id).write_text(
            json.dumps(asdict(candidate_plan), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _build_descriptor_and_artifacts(
        self,
        *,
        trial_id: str,
        parent_branch: str,
        parent_trial_id: str | None,
        candidate: CandidateMetadata,
    ):
        scratch_paths = self.worktree.cleanup_scratch_artifacts()
        touched = self.worktree.validate_changed_paths(
            allowed_paths=self.config.guardrails.allowed_paths,
            protected_paths=self.config.guardrails.protected_paths,
        )
        if not touched:
            raise WorktreeError("Candidate did not modify any tracked files")
        self._run_preflight()
        diff_text = self.worktree.diff_text(base_ref=parent_branch)
        artifact_dir = self._trial_artifact_dir(trial_id)
        candidate_path = self._artifact_candidate_path(trial_id)
        candidate_path.write_text(json.dumps(asdict(candidate), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        diff_path = artifact_dir / "patch.diff"
        diff_path.write_text(diff_text, encoding="utf-8")
        descriptor = build_patch_descriptor(
            touched_files=touched,
            diff_text=diff_text,
            candidate=candidate,
            parent_branch=parent_branch,
            parent_trial_id=parent_trial_id,
            budget_used="scout",
            diff_path=diff_path,
        )
        knob_summary = ""
        if descriptor.numeric_knobs:
            first_knob = next(iter(descriptor.numeric_knobs.items()))
            knob_summary = f" | knob={first_knob[0]}={first_knob[1]}"
        self._emit(
            kind="descriptor_ready",
            message=(
                f"Patch ready: files={len(descriptor.touched_files)}"
                f" | symbols={len(descriptor.touched_symbols)}"
                f" | family={descriptor.patch_category}/{descriptor.operation_type}"
                f"{knob_summary}"
            ),
            trial_id=trial_id,
            phase="execution",
        )
        if scratch_paths:
            self._emit(
                kind="execution_scratch_cleaned",
                message=f"Removed scratch artifacts before validation: {', '.join(scratch_paths[:3])}",
                trial_id=trial_id,
                phase="execution",
                metadata={"count": len(scratch_paths)},
            )
        return artifact_dir, diff_path, descriptor

    @staticmethod
    def _validate_informed_by_call_ids(call_ids: list[str], trace: list[SearchToolCall]) -> None:
        known = {item.call_id for item in trace}
        unknown = [call_id for call_id in call_ids if call_id not in known]
        if unknown:
            if not known:
                raise PolicyRejectedError(
                    "Agent cited BOA tool call ids, but BOA recorded no tool calls for this trial. "
                    "The planning or execution step likely hallucinated call ids instead of invoking `boa tools`."
                )
            raise PolicyRejectedError(f"Unknown BOA tool call ids referenced: {', '.join(unknown)}")

    def _validate_parent(self, *, candidate_plan: CandidatePlan, oracle: SearchOracleService) -> tuple[str, str | None]:
        lineage = oracle.validate_parent_selection(
            branch_name=candidate_plan.selected_parent_branch,
            trial_id=candidate_plan.selected_parent_trial_id,
        )
        if lineage is None:
            raise PolicyRejectedError(
                f"Selected parent is not a BOA-known lineage option: {candidate_plan.selected_parent_branch}"
            )
        return str(lineage["branch_name"]), None if lineage["trial_id"] is None else str(lineage["trial_id"])

    def _evaluate_candidate(
        self,
        *,
        trial_id: str,
        trial_branch: str,
        artifact_dir: Path,
    ) -> tuple[str, str | None, float | None, float | None]:
        canonical_stage: str | None = None
        canonical_score: float | None = None
        canonical_primary_metric: float | None = None
        final_accept = False
        highest_stage = self.acceptance.highest_enabled_stage()
        for stage_name in enabled_stages(self.config):
            self._emit(
                kind="stage_started",
                message=f"Starting evaluation stage '{stage_name}'",
                trial_id=trial_id,
                phase="evaluation",
                stage_name=stage_name,
            )
            execution = self.runner.run_stage(
                trial_id=trial_id,
                branch_name=trial_branch,
                worktree_path=self.worktree.worktree_path,
                stage_name=stage_name,
                stage=self._stage_config(stage_name),
                metrics=self.config.metrics,
                artifact_dir=artifact_dir,
            )
            if execution.status != "succeeded":
                stage_result = StageRunResult(
                    stage_name=stage_name,
                    branch_name=trial_branch,
                    status=execution.status,
                    command_results=execution.command_results,
                    metrics=execution.metrics,
                    primary_metric=None,
                    cost_metric=None,
                    adjusted_score=None,
                    threshold_passed=False,
                    improved=False,
                    advanced=False,
                    final_accept=False,
                    reason=execution.status,
                    started_at=execution.started_at,
                    completed_at=execution.completed_at,
                    resource_metadata=execution.resource_metadata,
                    artifact_dir=execution.artifact_dir,
                )
                self.store.record_stage_result(trial_id=trial_id, result=stage_result)
                self._emit(
                    kind="stage_failed",
                    message=f"Stage '{stage_name}' finished with status {execution.status}",
                    trial_id=trial_id,
                    phase="evaluation",
                    stage_name=stage_name,
                    metadata={"status": execution.status},
                )
                if execution.status == "timeout":
                    return "timed_out", canonical_stage, canonical_score, canonical_primary_metric
                if execution.status == "metric_missing":
                    return "metric_missing", canonical_stage, canonical_score, canonical_primary_metric
                return "stage_failed", canonical_stage, canonical_score, canonical_primary_metric
            incumbent = self.store.get_incumbent(stage_name)
            evaluation = self.acceptance.evaluate_stage(
                stage_name=stage_name,
                metrics=execution.metrics,
                incumbent=incumbent,
            )
            stage_result = StageRunResult(
                stage_name=stage_name,
                branch_name=trial_branch,
                status=execution.status,
                command_results=execution.command_results,
                metrics=execution.metrics,
                primary_metric=evaluation.primary_metric,
                cost_metric=evaluation.cost_metric,
                adjusted_score=evaluation.adjusted_score,
                threshold_passed=evaluation.threshold_passed,
                improved=evaluation.improved,
                advanced=evaluation.advanced,
                final_accept=evaluation.final_accept,
                reason=evaluation.reason,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                resource_metadata=execution.resource_metadata,
                artifact_dir=execution.artifact_dir,
            )
            self.store.record_stage_result(trial_id=trial_id, result=stage_result)
            canonical_stage = stage_name
            canonical_score = evaluation.adjusted_score
            canonical_primary_metric = evaluation.primary_metric
            final_accept = evaluation.final_accept
            self._emit(
                kind="stage_completed",
                message=(
                    f"Stage '{stage_name}' completed"
                    f" | score={evaluation.adjusted_score:.6f}"
                    f" | {self.config.objective.primary_metric}={evaluation.primary_metric:.6f}"
                ),
                trial_id=trial_id,
                phase="evaluation",
                stage_name=stage_name,
                metadata={
                    "score": evaluation.adjusted_score,
                    "primary_metric": evaluation.primary_metric,
                    "advanced": evaluation.advanced,
                    "branch_name": trial_branch,
                },
            )
            if evaluation.improved:
                self.store.upsert_incumbent(
                    stage_name=stage_name,
                    trial_id=trial_id,
                    branch_name=trial_branch,
                    adjusted_score=evaluation.adjusted_score,
                    primary_metric=evaluation.primary_metric,
                )
            if not evaluation.advanced:
                break
        if final_accept and canonical_stage == highest_stage and canonical_score is not None:
            return "completed_accepted", canonical_stage, canonical_score, canonical_primary_metric
        return "completed_rejected", canonical_stage, canonical_score, canonical_primary_metric

    @staticmethod
    def _classify_failure(exc: Exception) -> str:
        if isinstance(exc, ResearchAgentError):
            return "agent_failed"
        if isinstance(exc, WorktreeError):
            lowered = str(exc).lower()
            if "did not modify any tracked files" in lowered:
                return "no_patch"
            return "invalid_patch"
        if isinstance(exc, RunnerError):
            return "stage_failed"
        lowered = str(exc).lower()
        if "preflight command failed" in lowered:
            return "invalid_patch"
        if "missing primary metric" in lowered or "missing cost penalty metric" in lowered:
            return "metric_missing"
        if "timed out" in lowered:
            return "timed_out"
        return "stage_failed"

    @staticmethod
    def _summarize_exception_detail(exc: Exception) -> str | None:
        text = str(exc).strip()
        if not text:
            return None
        message_match = re.search(r'"message":\s*"((?:\\.|[^"])*)"', text, flags=re.DOTALL)
        if message_match:
            message = bytes(message_match.group(1), "utf-8").decode("unicode_escape")
            return " ".join(message.split())
        detail_match = re.search(r"Detail:\s*(.+)", text, flags=re.DOTALL)
        if detail_match:
            detail = detail_match.group(1).strip()
            if detail:
                return " ".join(detail.split())
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.lower() in {"stdout:", "stderr:", "<empty>"}:
                continue
            if line.startswith("```"):
                continue
            return " ".join(line.split())
        return None

    def _record_failed_trial(
        self,
        *,
        trial_id: str,
        trial_branch: str,
        parent_branch: str,
        parent_trial_id: str | None,
        candidate_plan: CandidatePlan | None,
        candidate: CandidateMetadata | None,
        descriptor,
        diff_path: Path | None,
        acceptance_status: str,
        exc: Exception,
    ) -> None:
        artifact_dir = self._trial_artifact_dir(trial_id)
        (artifact_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        trace = load_search_trace(self._search_trace_path(trial_id))
        failure_parent_branch = candidate_plan.selected_parent_branch if candidate_plan else parent_branch
        failure_parent_trial_id = candidate_plan.selected_parent_trial_id if candidate_plan else parent_trial_id
        if self.store.has_trial(trial_id):
            self.store.set_acceptance(
                trial_id=trial_id,
                acceptance_status=acceptance_status,
                canonical_stage=None,
                canonical_score=None,
            )
            return
        self.store.create_trial(
            run_tag=self.config.run.tag,
            trial_id=trial_id,
            branch_name=trial_branch,
            parent_branch=failure_parent_branch,
            parent_trial_id=failure_parent_trial_id,
            candidate_plan=candidate_plan,
            candidate=candidate,
            descriptor=descriptor,
            search_trace=trace,
            diff_path=diff_path,
            acceptance_status=acceptance_status,
        )

    def run(self) -> RunSummary:
        self._emit(
            kind="run_started",
            message=f"Starting BOA run for tag '{self.config.run.tag}'",
        )
        self._ensure_ready()
        consecutive_failures = 0
        trials_attempted = 0
        last_trial_id: str | None = None
        last_acceptance_status: str | None = None
        last_canonical_stage: str | None = None
        last_canonical_score: float | None = None
        last_detail: str | None = None
        for _ in range(int(self.config.run.max_trials)):
            if self.paths.stop_file.exists():
                self._emit(
                    kind="run_stopped",
                    message="Stop file detected. Ending BOA run.",
                )
                return RunSummary(
                    trials_attempted=trials_attempted,
                    stop_requested=True,
                    last_trial_id=last_trial_id,
                    last_acceptance_status=last_acceptance_status,
                    last_canonical_stage=last_canonical_stage,
                    last_canonical_score=last_canonical_score,
                    last_detail=None,
                )
            for stage_name in enabled_stages(self.config):
                self._seed_baseline(stage_name)
            recent_trials = self._recent_trials()
            oracle = self._search_oracle(recent_trials)
            trial_id = self._next_trial_id()
            trial_branch = self._trial_branch(trial_id)
            parent_branch = str(self.config.run.accepted_branch)
            parent_trial_id: str | None = None
            candidate_plan: CandidatePlan | None = None
            candidate: CandidateMetadata | None = None
            descriptor = None
            diff_path: Path | None = None
            trials_attempted += 1
            last_trial_id = trial_id
            self._emit(
                kind="trial_started",
                message=f"Starting trial {trial_id}",
                trial_id=trial_id,
            )
            try:
                self._prepare_planning_workspace()
                self._emit(
                    kind="planning_started",
                    message="Preparing planning phase on accepted branch",
                    trial_id=trial_id,
                    phase="planning",
                )
                planning_context = self._build_planning_context(trial_id=trial_id, recent_trials=recent_trials)
                candidate_plan = self.agent.plan_trial(planning_context)
                self._emit(
                    kind="planning_completed",
                    message=f"Planning selected parent branch {candidate_plan.selected_parent_branch}",
                    trial_id=trial_id,
                    phase="planning",
                    metadata={
                        "parent_branch": candidate_plan.selected_parent_branch,
                        "parent_trial_id": candidate_plan.selected_parent_trial_id,
                        "patch_category": candidate_plan.patch_category,
                        "operation_type": candidate_plan.operation_type,
                    },
                )
                self._write_plan_artifact(trial_id=trial_id, candidate_plan=candidate_plan)
                # Write plan.json from the BOA process so the execution agent can read it,
                # regardless of whether the planning agent could write it directly.
                plan_out = self._plan_output_path(trial_id)
                plan_out.parent.mkdir(parents=True, exist_ok=True)
                plan_out.write_text(
                    json.dumps(asdict(candidate_plan), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self._validate_informed_by_call_ids(
                    candidate_plan.informed_by_call_ids,
                    load_search_trace(self._search_trace_path(trial_id)),
                )
                parent_branch, parent_trial_id = self._validate_parent(candidate_plan=candidate_plan, oracle=oracle)
                self.worktree.prepare_trial(trial_branch=trial_branch, parent_branch=parent_branch)
                self._emit(
                    kind="execution_started",
                    message=f"Preparing execution worktree from {parent_branch}",
                    trial_id=trial_id,
                    phase="execution",
                    metadata={
                        "parent_branch": parent_branch,
                        "parent_trial_id": parent_trial_id,
                        "branch_name": trial_branch,
                    },
                )
                execution_context = self._build_execution_context(
                    trial_id=trial_id,
                    trial_branch=trial_branch,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                    recent_trials=recent_trials,
                    candidate_plan=candidate_plan,
                )
                candidate = self.agent.prepare_candidate(execution_context)
                self._emit(
                    kind="execution_completed",
                    message="Candidate metadata captured. Validating patch and running stages.",
                    trial_id=trial_id,
                    phase="execution",
                )
                self._validate_informed_by_call_ids(
                    candidate.informed_by_call_ids,
                    load_search_trace(self._search_trace_path(trial_id)),
                )
                artifact_dir, diff_path, descriptor = self._build_descriptor_and_artifacts(
                    trial_id=trial_id,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                    candidate=candidate,
                )
                self.store.create_trial(
                    run_tag=self.config.run.tag,
                    trial_id=trial_id,
                    branch_name=trial_branch,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                    candidate_plan=candidate_plan,
                    candidate=candidate,
                    descriptor=descriptor,
                    search_trace=load_search_trace(self._search_trace_path(trial_id)),
                    diff_path=diff_path,
                    acceptance_status="running",
                )
                self.worktree.commit_trial(self._commit_message(trial_id, candidate))
                if self.runner.requires_remote_branches:
                    self.worktree.push_branch(remote=self.config.runner.ssh.git_remote, branch=trial_branch, force=True)
                acceptance_status, canonical_stage, canonical_score, canonical_primary_metric = self._evaluate_candidate(
                    trial_id=trial_id,
                    trial_branch=trial_branch,
                    artifact_dir=artifact_dir,
                )
                if acceptance_status == "completed_accepted":
                    self.worktree.promote_trial(trial_branch=trial_branch)
                    if self.runner.requires_remote_branches:
                        self.worktree.push_branch(
                            remote=self.config.runner.ssh.git_remote,
                            branch=str(self.config.run.accepted_branch),
                            force=False,
                        )
                self.store.set_acceptance(
                    trial_id=trial_id,
                    acceptance_status=acceptance_status,
                    canonical_stage=canonical_stage,
                    canonical_score=canonical_score,
                )
                last_acceptance_status = acceptance_status
                last_canonical_stage = canonical_stage
                last_canonical_score = canonical_score
                last_detail = None
                consecutive_failures = 0
                self._emit(
                    kind="trial_completed",
                    message=f"Trial {trial_id} finished with status {acceptance_status}",
                    trial_id=trial_id,
                    metadata={
                        "acceptance_status": acceptance_status,
                        "branch_name": trial_branch,
                        "parent_branch": parent_branch,
                        "parent_trial_id": parent_trial_id,
                        "canonical_stage": canonical_stage,
                        "canonical_score": canonical_score,
                        "primary_metric": canonical_primary_metric,
                    },
                )
            except PolicyRejectedError as exc:
                consecutive_failures += 1
                last_acceptance_status = "policy_rejected"
                last_canonical_stage = None
                last_canonical_score = None
                last_detail = self._summarize_exception_detail(exc)
                self._record_failed_trial(
                    trial_id=trial_id,
                    trial_branch=trial_branch,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                    candidate_plan=candidate_plan,
                    candidate=candidate,
                    descriptor=descriptor,
                    diff_path=diff_path,
                    acceptance_status="policy_rejected",
                    exc=exc,
                )
                self._emit(
                    kind="trial_failed",
                    message=f"Trial {trial_id} was policy rejected",
                    trial_id=trial_id,
                    metadata={
                        "acceptance_status": "policy_rejected",
                        "detail": last_detail or "",
                        "branch_name": trial_branch,
                        "parent_branch": parent_branch,
                        "parent_trial_id": parent_trial_id,
                    },
                )
                if consecutive_failures >= int(self.config.run.max_consecutive_failures):
                    raise
            except (ResearchAgentError, RunnerError, WorktreeError, RuntimeError, ValueError) as exc:
                consecutive_failures += 1
                last_acceptance_status = self._classify_failure(exc)
                last_canonical_stage = None
                last_canonical_score = None
                last_detail = self._summarize_exception_detail(exc)
                self._record_failed_trial(
                    trial_id=trial_id,
                    trial_branch=trial_branch,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                    candidate_plan=candidate_plan,
                    candidate=candidate,
                    descriptor=descriptor,
                    diff_path=diff_path,
                    acceptance_status=last_acceptance_status,
                    exc=exc,
                )
                self._emit(
                    kind="trial_failed",
                    message=f"Trial {trial_id} failed with status {last_acceptance_status}",
                    trial_id=trial_id,
                    metadata={
                        "acceptance_status": last_acceptance_status or "unknown",
                        "detail": last_detail or "",
                        "branch_name": trial_branch,
                        "parent_branch": parent_branch,
                        "parent_trial_id": parent_trial_id,
                    },
                )
                if consecutive_failures >= int(self.config.run.max_consecutive_failures):
                    raise
        summary = RunSummary(
            trials_attempted=trials_attempted,
            stop_requested=False,
            last_trial_id=last_trial_id,
            last_acceptance_status=last_acceptance_status,
            last_canonical_stage=last_canonical_stage,
            last_canonical_score=last_canonical_score,
            last_detail=last_detail,
        )
        self._emit(
            kind="run_completed",
            message="BOA run completed",
            trial_id=last_trial_id,
            metadata={"acceptance_status": last_acceptance_status or "unknown"},
        )
        return summary
