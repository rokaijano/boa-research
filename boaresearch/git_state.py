from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


class GitError(RuntimeError):
    pass


def run_git(
    repo_path: Path,
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=capture_output,
        text=True,
        env=merged_env,
    )
    if check and proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise GitError(f"git {' '.join(args)} failed: {stderr}")
    return proc


def current_branch(repo_path: Path) -> str:
    return run_git(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def current_commit(repo_path: Path) -> str:
    return run_git(repo_path, ["rev-parse", "HEAD"]).stdout.strip()


def resolve_ref(repo_path: Path, ref: str) -> str:
    return run_git(repo_path, ["rev-parse", ref]).stdout.strip()


def short_commit(commit: str, length: int = 12) -> str:
    return str(commit).strip()[:length]


def branch_exists(repo_path: Path, branch: str) -> bool:
    proc = run_git(repo_path, ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False)
    return proc.returncode == 0


def ensure_branch(repo_path: Path, branch: str, start_point: str) -> None:
    if branch_exists(repo_path, branch):
        return
    run_git(repo_path, ["branch", branch, start_point], capture_output=True)


def force_branch(repo_path: Path, branch: str, start_point: str) -> None:
    run_git(repo_path, ["branch", "-f", branch, start_point], capture_output=True)


def add_worktree(repo_path: Path, worktree_path: Path, branch: str, start_point: str) -> None:
    if (worktree_path / ".git").exists():
        return
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    run_git(
        repo_path,
        ["worktree", "add", "-B", branch, str(worktree_path), start_point],
        capture_output=True,
    )


def prune_worktrees(repo_path: Path) -> None:
    run_git(repo_path, ["worktree", "prune"], capture_output=True)


def checkout(repo_path: Path, branch: str) -> None:
    run_git(repo_path, ["checkout", branch], capture_output=True)


def hard_reset(repo_path: Path, target: str) -> None:
    run_git(repo_path, ["reset", "--hard", target], capture_output=True)


def clean_untracked(repo_path: Path) -> None:
    run_git(repo_path, ["clean", "-fd"], capture_output=True)


def apply_patch(repo_path: Path, patch_path: Path) -> None:
    run_git(
        repo_path,
        ["apply", "--whitespace=nowarn", "--ignore-space-change", "--ignore-whitespace", str(patch_path)],
        capture_output=True,
    )


def status_porcelain(repo_path: Path) -> list[str]:
    out = run_git(repo_path, ["status", "--porcelain=v1"]).stdout
    return [line for line in out.splitlines() if line.strip()]


def changed_paths(repo_path: Path) -> list[str]:
    touched: list[str] = []
    for line in status_porcelain(repo_path):
        candidate = line[3:].strip()
        if " -> " in candidate:
            candidate = candidate.split(" -> ", 1)[1]
        candidate = candidate.replace("\\", "/")
        if candidate and candidate not in touched:
            touched.append(candidate)
    return touched


def diff_text(repo_path: Path, *, base_ref: str | None = None) -> str:
    args = ["diff", "--binary"]
    if base_ref:
        args.append(base_ref)
    return run_git(repo_path, args).stdout


def changed_paths_against(repo_path: Path, base_ref: str) -> list[str]:
    proc = run_git(repo_path, ["diff", "--name-only", base_ref], capture_output=True)
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def remove_branch(repo_path: Path, branch: str) -> None:
    run_git(repo_path, ["branch", "-D", branch], check=False, capture_output=True)


def commit_all(repo_path: Path, message: str) -> str:
    if not status_porcelain(repo_path):
        return current_commit(repo_path)
    run_git(repo_path, ["add", "-A"], capture_output=True)
    run_git(repo_path, ["commit", "-m", message], capture_output=True)
    return current_commit(repo_path)


def push_branch(repo_path: Path, remote: str, branch: str, *, force: bool = False, env: Optional[dict[str, str]] = None) -> None:
    args = ["push"]
    if force:
        args.append("--force-with-lease")
    args.extend([remote, f"{branch}:{branch}"])
    run_git(repo_path, args, capture_output=True, env=env)
