from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema import BoaConfig


@dataclass(frozen=True)
class BoaPaths:
    repo_root: Path
    runtime_root: Path
    artifacts_root: Path
    trials_root: Path
    store_root: Path
    store_path: Path
    prompts_root: Path
    worktrees_root: Path
    stop_file: Path

    @classmethod
    def from_config(cls, cfg: BoaConfig) -> "BoaPaths":
        runtime_root = (cfg.repo_root / ".boa").resolve()
        return cls(
            repo_root=cfg.repo_root,
            runtime_root=runtime_root,
            artifacts_root=(runtime_root / "artifacts").resolve(),
            trials_root=(runtime_root / "artifacts" / "trials").resolve(),
            store_root=(runtime_root / "store").resolve(),
            store_path=(runtime_root / "store" / "experiments.sqlite").resolve(),
            prompts_root=(runtime_root / "prompts").resolve(),
            worktrees_root=(runtime_root / "worktrees").resolve(),
            stop_file=(cfg.run.stop_file or (runtime_root / "STOP")).resolve(),
        )

    def ensure(self) -> None:
        for path in [
            self.runtime_root,
            self.artifacts_root,
            self.trials_root,
            self.store_root,
            self.prompts_root,
            self.worktrees_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def trial_artifact_dir(self, trial_id: str) -> Path:
        return (self.trials_root / trial_id).resolve()

    def prompt_bundle_dir(self, trial_id: str) -> Path:
        return (self.prompts_root / trial_id).resolve()
