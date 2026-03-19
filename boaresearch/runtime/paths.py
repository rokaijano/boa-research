from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema import BoaConfig


@dataclass(frozen=True)
class BoaPaths:
    repo_root: Path
    runtime_root: Path
    protected_root: Path
    worktree_root: Path
    artifacts_root: Path
    trials_root: Path
    store_root: Path
    store_path: Path
    prompts_root: Path
    agent_outputs_root: Path
    agent_traces_root: Path
    worktrees_root: Path
    stop_file: Path

    @classmethod
    def from_config(cls, cfg: BoaConfig) -> "BoaPaths":
        runtime_root = (cfg.repo_root / ".boa").resolve()
        protected_root = (runtime_root / "protected").resolve()
        worktree_root = (runtime_root / "worktree").resolve()
        return cls(
            repo_root=cfg.repo_root,
            runtime_root=runtime_root,
            protected_root=protected_root,
            worktree_root=worktree_root,
            artifacts_root=(protected_root / "artifacts").resolve(),
            trials_root=(protected_root / "artifacts" / "trials").resolve(),
            store_root=(protected_root / "store").resolve(),
            store_path=(protected_root / "store" / "experiments.sqlite").resolve(),
            prompts_root=(protected_root / "prompts").resolve(),
            agent_outputs_root=(protected_root / "agent_outputs").resolve(),
            agent_traces_root=(protected_root / "agent_traces").resolve(),
            worktrees_root=worktree_root,
            stop_file=(cfg.run.stop_file or (protected_root / "STOP")).resolve(),
        )

    def ensure(self) -> None:
        for path in [
            self.runtime_root,
            self.protected_root,
            self.worktree_root,
            self.artifacts_root,
            self.trials_root,
            self.store_root,
            self.prompts_root,
            self.agent_outputs_root,
            self.agent_traces_root,
            self.worktrees_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def trial_artifact_dir(self, trial_id: str) -> Path:
        return (self.trials_root / trial_id).resolve()

    def prompt_bundle_dir(self, trial_id: str) -> Path:
        return (self.prompts_root / trial_id).resolve()

    def agent_output_dir(self, trial_id: str) -> Path:
        return (self.agent_outputs_root / trial_id).resolve()

    def search_trace_path(self, trial_id: str) -> Path:
        return (self.agent_traces_root / trial_id / "search_calls.jsonl").resolve()

    def worktree_dir(self, run_tag: str) -> Path:
        return (self.worktrees_root / run_tag).resolve()
