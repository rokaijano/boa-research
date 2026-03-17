from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from . import git_state
from .acceptance import AcceptanceEngine
from .agents import ResearchAgentError, build_agent
from .descriptors import build_patch_descriptor
from .git_auth import GitAuthManager
from .loader import enabled_stages
from .paths import BoaPaths
from .runner import RunnerError, build_trial_runner
from .schema import AgentContext, BoaConfig, StageRunResult, TrialSummary
from .search import RepoSearchState, build_search_policy
from .store import ExperimentStore
from .worktree import WorktreeError, WorktreeManager


class BoaController:
    def __init__(self, config: BoaConfig) -> None:
        self.config = config
        self.paths = BoaPaths.from_config(config)
        helper_root = self.paths.runtime_root / "git_auth_helpers"
        self.git_auth = GitAuthManager(config.git_auth, helper_root=helper_root)
        self.store = ExperimentStore(self.paths.store_path)
        self.acceptance = AcceptanceEngine(config)
        self.search_policy = build_search_policy(config)
        self.runner = build_trial_runner(config, git_auth=self.git_auth)
        self.worktree = WorktreeManager(
            repo_root=config.repo_root,
            worktree_path=config.run.worktree_path or (config.repo_root / ".boa" / "worktrees" / config.run.tag),
            accepted_branch=str(config.run.accepted_branch),
            git_auth=self.git_auth,
        )
        self.agent = build_agent(
            config,
            repo_root=config.repo_root,
            run_preflight=self._run_preflight,
            read_recent_trials=self._read_recent_trials,
        )
        self._trial_counter = 0

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

    def _read_recent_trials(self, limit: int = 10) -> list[dict[str, str]]:
        rows = []
        for trial in self.store.recent_trials(limit=limit):
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "acceptance_status": trial.acceptance_status,
                    "canonical_stage": trial.canonical_stage or "",
                    "canonical_score": "" if trial.canonical_score is None else f"{trial.canonical_score:.6f}",
                    "patch_category": trial.descriptor.patch_category if trial.descriptor else "",
                    "summary": trial.candidate.rationale_summary if trial.candidate else "",
                }
            )
        return rows

    def _run_command(self, *, command: str, cwd: Path) -> None:
        args = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]
        proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Preflight command failed: {command}\n{detail}")

    def _run_preflight(self) -> None:
        for command in self.config.guardrails.preflight_commands:
            self._run_command(command=command, cwd=self.worktree.worktree_path)

    def _ensure_ready(self) -> None:
        self.paths.ensure()
        self.store.ensure_schema()
        if self.config.run.base_branch is None:
            self.config.run.base_branch = git_state.current_branch(self.config.repo_root)
        self.worktree.ensure_accepted_branch(base_branch=str(self.config.run.base_branch))
        if self.runner.requires_remote_branches:
            self.worktree.push_branch(
                remote=self.config.runner.ssh.git_remote,
                branch=str(self.config.run.accepted_branch),
                force=False,
            )

    def _next_trial_id(self) -> str:
        self._trial_counter += 1
        return f"{self.config.run.tag}-{self._trial_counter:04d}"

    def _trial_branch(self, trial_id: str) -> str:
        return f"boa/{self.config.run.tag}/trial/{trial_id}"

    def _trial_artifact_dir(self, trial_id: str) -> Path:
        path = self.paths.trial_artifact_dir(trial_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _commit_message(self, trial_id: str, candidate) -> str:
        return f"boa: {candidate.patch_category} trial {trial_id}"

    def _build_agent_context(self, *, trial_id: str, trial_branch: str, search_decision, recent_trials: list[TrialSummary]) -> AgentContext:
        return AgentContext(
            repo_root=self.config.repo_root,
            worktree_path=self.worktree.worktree_path,
            trial_id=trial_id,
            run_tag=self.config.run.tag,
            accepted_branch=str(self.config.run.accepted_branch),
            trial_branch=trial_branch,
            boa_md_path=self.config.boa_md_path,
            extra_context_files=list(self.config.agent.extra_context_files),
            allowed_paths=list(self.config.guardrails.allowed_paths),
            protected_paths=list(self.config.guardrails.protected_paths),
            search_decision=search_decision,
            recent_trials=recent_trials,
            objective_summary=self._objective_summary(),
            preflight_commands=list(self.config.guardrails.preflight_commands),
            max_agent_steps=int(self.config.agent.max_agent_steps),
            prompt_bundle_dir=self.paths.prompt_bundle_dir(trial_id),
        )

    def _stage_config(self, stage_name: str):
        return getattr(self.config.runner, stage_name)

    def _seed_baseline(self, stage_name: str) -> None:
        if self.store.get_incumbent(stage_name) is not None:
            return
        trial_id = f"{self.config.run.tag}-baseline-{stage_name}"
        artifact_dir = self._trial_artifact_dir(trial_id)
        if not self.runner.requires_remote_branches:
            self.worktree.prepare_trial(
                trial_branch=str(self.config.run.accepted_branch),
                parent_branch=str(self.config.run.accepted_branch),
            )
        self.store.create_trial(
            run_tag=self.config.run.tag,
            trial_id=trial_id,
            branch_name=str(self.config.run.accepted_branch),
            parent_branch=str(self.config.run.accepted_branch),
            parent_trial_id=None,
            candidate=None,
            descriptor=None,
            search_decision={"policy": "baseline", "stage": stage_name},
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
        evaluation = self.acceptance.evaluate_stage(stage_name=stage_name, metrics=execution.metrics, incumbent=None)
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

    def _build_descriptor_and_artifacts(self, *, trial_id: str, parent_branch: str, parent_trial_id: str | None, candidate):
        touched = self.worktree.validate_changed_paths(
            allowed_paths=self.config.guardrails.allowed_paths,
            protected_paths=self.config.guardrails.protected_paths,
        )
        if not touched:
            raise WorktreeError("Candidate did not modify any tracked files")
        self._run_preflight()
        diff_text = self.worktree.diff_text(base_ref=parent_branch)
        artifact_dir = self._trial_artifact_dir(trial_id)
        candidate_path = artifact_dir / "candidate.json"
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
        return artifact_dir, diff_path, descriptor

    def _evaluate_candidate(
        self,
        *,
        trial_id: str,
        trial_branch: str,
        artifact_dir: Path,
    ) -> tuple[str, str | None, float | None]:
        canonical_stage: str | None = None
        canonical_score: float | None = None
        highest_stage = self.acceptance.highest_enabled_stage()
        for stage_name in enabled_stages(self.config):
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
                break
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
            if evaluation.improved:
                self.store.upsert_incumbent(
                    stage_name=stage_name,
                    trial_id=trial_id,
                    branch_name=trial_branch,
                    adjusted_score=evaluation.adjusted_score,
                    primary_metric=evaluation.primary_metric,
                )
            if evaluation.advanced:
                canonical_stage = stage_name
                canonical_score = evaluation.adjusted_score
            if not evaluation.advanced:
                break
        if canonical_stage == highest_stage and canonical_score is not None:
            return "accepted", canonical_stage, canonical_score
        if canonical_stage is not None:
            return "staged_only", canonical_stage, canonical_score
        return "rejected", None, None

    def run(self) -> None:
        self._ensure_ready()
        consecutive_failures = 0
        for _ in range(int(self.config.run.max_trials)):
            if self.paths.stop_file.exists():
                return
            for stage_name in enabled_stages(self.config):
                self._seed_baseline(stage_name)
            recent_trials = self.store.recent_trials(limit=self.config.search.max_history)
            search_decision = self.search_policy.propose(
                recent_trials,
                RepoSearchState(accepted_branch=str(self.config.run.accepted_branch)),
                self.config,
            )
            trial_id = self._next_trial_id()
            trial_branch = self._trial_branch(trial_id)
            parent_branch = search_decision.parent_branch or str(self.config.run.accepted_branch)
            try:
                self.worktree.prepare_trial(trial_branch=trial_branch, parent_branch=parent_branch)
                context = self._build_agent_context(
                    trial_id=trial_id,
                    trial_branch=trial_branch,
                    search_decision=search_decision,
                    recent_trials=recent_trials,
                )
                candidate = self.agent.prepare_candidate(context)
                artifact_dir, diff_path, descriptor = self._build_descriptor_and_artifacts(
                    trial_id=trial_id,
                    parent_branch=parent_branch,
                    parent_trial_id=search_decision.parent_trial_id,
                    candidate=candidate,
                )
                self.store.create_trial(
                    run_tag=self.config.run.tag,
                    trial_id=trial_id,
                    branch_name=trial_branch,
                    parent_branch=parent_branch,
                    parent_trial_id=search_decision.parent_trial_id,
                    candidate=candidate,
                    descriptor=descriptor,
                    search_decision=asdict(search_decision),
                    diff_path=diff_path,
                    acceptance_status="running",
                )
                self.worktree.commit_trial(self._commit_message(trial_id, candidate))
                if self.runner.requires_remote_branches:
                    self.worktree.push_branch(remote=self.config.runner.ssh.git_remote, branch=trial_branch, force=True)
                acceptance_status, canonical_stage, canonical_score = self._evaluate_candidate(
                    trial_id=trial_id,
                    trial_branch=trial_branch,
                    artifact_dir=artifact_dir,
                )
                if acceptance_status == "accepted":
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
                consecutive_failures = 0
            except (ResearchAgentError, RunnerError, WorktreeError, RuntimeError, ValueError) as exc:
                consecutive_failures += 1
                failure_artifact_dir = self._trial_artifact_dir(trial_id)
                (failure_artifact_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                if self.store.has_trial(trial_id):
                    self.store.set_acceptance(
                        trial_id=trial_id,
                        acceptance_status="failed",
                        canonical_stage=None,
                        canonical_score=None,
                    )
                else:
                    self.store.create_trial(
                        run_tag=self.config.run.tag,
                        trial_id=trial_id,
                        branch_name=trial_branch,
                        parent_branch=parent_branch,
                        parent_trial_id=search_decision.parent_trial_id,
                        candidate=None,
                        descriptor=None,
                        search_decision=asdict(search_decision),
                        diff_path=None,
                        acceptance_status="failed",
                    )
                if consecutive_failures >= int(self.config.run.max_consecutive_failures):
                    raise
