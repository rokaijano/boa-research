from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional

from . import git_state
from .git_auth import GitAuthManager


class WorktreeError(RuntimeError):
    pass


def _normalize_repo_path(path: str) -> str:
    normalized = str(PurePosixPath(str(path).replace("\\", "/"))).lstrip("./")
    return normalized


def extract_patch_paths(patch_text: str) -> list[str]:
    touched: list[str] = []
    for raw_line in str(patch_text).splitlines():
        if raw_line.startswith("+++ "):
            candidate = raw_line[4:].strip()
            if candidate == "/dev/null":
                continue
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            normalized = _normalize_repo_path(candidate)
            if normalized and normalized not in touched:
                touched.append(normalized)
    return touched


def _path_matches_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def validate_patch_paths(paths: list[str], allowed_paths: list[str], protected_paths: list[str]) -> None:
    for path in paths:
        normalized = _normalize_repo_path(path)
        parts = PurePosixPath(normalized).parts
        if os.path.isabs(normalized) or ".." in parts:
            raise WorktreeError(f"Patch touched an invalid path: {path}")
        if protected_paths and any(_path_matches_prefix(normalized, prefix) for prefix in protected_paths):
            raise WorktreeError(f"Patch touched a protected path: {path}")
        if allowed_paths and not any(_path_matches_prefix(normalized, prefix) for prefix in allowed_paths):
            raise WorktreeError(f"Patch touched a path outside guardrails.allowed_paths: {path}")


@dataclass
class WorktreeManager:
    repo_root: Path
    worktree_path: Path
    accepted_branch: str
    git_auth: Optional[GitAuthManager] = None

    def ensure_accepted_branch(self, *, base_branch: str) -> None:
        git_state.prune_worktrees(self.repo_root)
        git_state.ensure_branch(self.repo_root, self.accepted_branch, base_branch)

    def prepare_trial(self, *, trial_branch: str, parent_branch: str) -> None:
        git_state.prune_worktrees(self.repo_root)
        self.worktree_path.parent.mkdir(parents=True, exist_ok=True)
        if not (self.worktree_path / ".git").exists():
            git_state.add_worktree(self.repo_root, self.worktree_path, trial_branch, parent_branch)
        git_state.checkout(self.worktree_path, parent_branch)
        git_state.run_git(self.worktree_path, ["checkout", "-B", trial_branch, parent_branch], capture_output=True)
        git_state.hard_reset(self.worktree_path, parent_branch)
        git_state.clean_untracked(self.worktree_path)

    def changed_paths(self) -> list[str]:
        return [_normalize_repo_path(path) for path in git_state.changed_paths(self.worktree_path)]

    def diff_text(self, *, base_ref: str) -> str:
        return git_state.diff_text(self.worktree_path, base_ref=base_ref)

    def validate_changed_paths(self, *, allowed_paths: list[str], protected_paths: list[str]) -> list[str]:
        touched = self.changed_paths()
        validate_patch_paths(touched, allowed_paths=allowed_paths, protected_paths=protected_paths)
        return touched

    def commit_trial(self, message: str) -> str:
        return git_state.commit_all(self.worktree_path, message)

    def promote_trial(self, *, trial_branch: str) -> str:
        git_state.force_branch(self.repo_root, self.accepted_branch, trial_branch)
        return git_state.resolve_ref(self.repo_root, self.accepted_branch)

    def accepted_commit(self) -> str:
        return git_state.resolve_ref(self.repo_root, self.accepted_branch)

    def current_branch(self) -> str:
        return git_state.current_branch(self.worktree_path)

    def apply_patch(self, patch_text: str, *, allowed_paths: list[str], protected_paths: list[str]) -> list[str]:
        if not str(patch_text).strip():
            return []
        touched = extract_patch_paths(patch_text)
        validate_patch_paths(touched, allowed_paths=allowed_paths, protected_paths=protected_paths)
        patch_fd, patch_name = tempfile.mkstemp(prefix="boa_", suffix=".patch", dir=str(self.worktree_path))
        os.close(patch_fd)
        patch_path = Path(patch_name)
        try:
            patch_path.write_text(str(patch_text), encoding="utf-8", newline="\n")
            git_state.apply_patch(self.worktree_path, patch_path)
        finally:
            try:
                patch_path.unlink()
            except OSError:
                pass
        return touched

    @contextmanager
    def _push_env(self) -> Iterator[dict[str, str]]:
        if self.git_auth is None:
            yield {}
            return
        with self.git_auth.local_git_env(enabled=self.git_auth.config.use_for_local_push) as env:
            yield env

    def push_branch(self, *, remote: str, branch: str, force: bool) -> None:
        with self._push_env() as env:
            git_state.push_branch(self.repo_root, remote, branch, force=force, env=env)
