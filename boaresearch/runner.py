from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .git_auth import GitAuthManager
from .metrics import extract_metrics
from .schema import LocalRunnerConfig, MetricConfig, RunnerStageConfig, SSHRunnerConfig, StageCommandResult


class RunnerError(RuntimeError):
    pass


@dataclass
class StageExecution:
    stage_name: str
    branch_name: str
    status: str
    command_results: list[StageCommandResult]
    metrics: dict[str, float]
    resource_metadata: dict[str, object]
    artifact_dir: Path
    started_at: str
    completed_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_epoch(raw: str | None) -> str:
    try:
        value = float(raw or "")
    except ValueError:
        return utc_now()
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _shell_args(command: str) -> list[str]:
    return ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]


class BaseTrialRunner(ABC):
    requires_remote_branches = False

    @abstractmethod
    def run_stage(
        self,
        *,
        trial_id: str,
        branch_name: str,
        worktree_path: Path,
        stage_name: str,
        stage: RunnerStageConfig,
        metrics: list[MetricConfig],
        artifact_dir: Path,
    ) -> StageExecution:
        raise NotImplementedError


class LocalTrialRunner(BaseTrialRunner):
    def __init__(self, config: LocalRunnerConfig) -> None:
        self.config = config

    def _command_script(self, command: str) -> str:
        if self.config.activation_command:
            return f"{self.config.activation_command}\n{command}"
        return command

    def run_stage(
        self,
        *,
        trial_id: str,
        branch_name: str,
        worktree_path: Path,
        stage_name: str,
        stage: RunnerStageConfig,
        metrics: list[MetricConfig],
        artifact_dir: Path,
    ) -> StageExecution:
        del trial_id
        stage_artifact_dir = (artifact_dir / stage_name).resolve()
        stage_artifact_dir.mkdir(parents=True, exist_ok=True)
        started_at = utc_now()
        command_results: list[StageCommandResult] = []
        env = os.environ.copy()
        env.update(self.config.env)
        env.update(stage.env)
        status = "succeeded"
        for index, command in enumerate(stage.commands, start=1):
            stdout_path = stage_artifact_dir / f"command_{index:03d}.stdout"
            stderr_path = stage_artifact_dir / f"command_{index:03d}.stderr"
            command_started_at = datetime.now(timezone.utc)
            try:
                proc = subprocess.run(
                    _shell_args(self._command_script(command)),
                    cwd=str(worktree_path),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    timeout=int(stage.timeout_seconds),
                )
                stdout_path.write_text(proc.stdout or "", encoding="utf-8")
                stderr_path.write_text(proc.stderr or "", encoding="utf-8")
                command_status = "ok" if proc.returncode == 0 else "failed"
                if proc.returncode != 0:
                    status = "failed"
            except subprocess.TimeoutExpired as exc:
                stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
                stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "", encoding="utf-8")
                proc = None
                command_status = "timeout"
                status = "timeout"
            command_finished_at = datetime.now(timezone.utc)
            command_results.append(
                StageCommandResult(
                    index=index,
                    command=command,
                    exit_code=124 if command_status == "timeout" else int(proc.returncode if proc is not None else 124),
                    status=command_status,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    wall_time_seconds=max(0.0, (command_finished_at - command_started_at).total_seconds()),
                    max_rss_kb=None,
                )
            )
            if command_status != "ok":
                break
        if status == "succeeded":
            captured_root = stage_artifact_dir / "captured"
            for metric in metrics:
                if not metric.path:
                    continue
                source_path = worktree_path / metric.path
                if not source_path.exists():
                    continue
                target_path = captured_root / metric.path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
        extracted_metrics = extract_metrics(artifact_dir=stage_artifact_dir, metrics=metrics) if status == "succeeded" else {}
        return StageExecution(
            stage_name=stage_name,
            branch_name=branch_name,
            status=status,
            command_results=command_results,
            metrics=extracted_metrics,
            resource_metadata={"execution_mode": "local"},
            artifact_dir=stage_artifact_dir,
            started_at=started_at,
            completed_at=utc_now(),
        )


class SSHTrialRunner(BaseTrialRunner):
    requires_remote_branches = True

    def __init__(self, config: SSHRunnerConfig, *, git_auth: Optional[GitAuthManager] = None) -> None:
        self.config = config
        self.git_auth = git_auth

    def _target(self) -> str:
        if self.config.host_alias:
            return self.config.host_alias
        if self.config.user:
            return f"{self.config.user}@{self.config.host}"
        return str(self.config.host)

    def _ssh_base_args(self) -> list[str]:
        args = ["ssh"]
        if self.config.port:
            args.extend(["-p", str(self.config.port)])
        if self.config.key_path is not None:
            args.extend(["-i", str(self.config.key_path)])
        args.append(self._target())
        return args

    def _run_remote_script(self, script: str, *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self._ssh_base_args(), "bash", "-lc", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )

    def _fetch_stage_dir(self, *, remote_stage_dir: str, local_stage_dir: Path) -> None:
        local_stage_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                *self._ssh_base_args(),
                "bash",
                "-lc",
                f"cd {shlex.quote(self.config.repo_path)} && tar -C {shlex.quote(remote_stage_dir)} -cf - .",
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RunnerError(f"Failed to fetch remote stage artifacts: {detail}")
        extract = subprocess.run(
            ["tar", "-xf", "-", "-C", str(local_stage_dir)],
            input=proc.stdout,
            capture_output=True,
            check=False,
        )
        if extract.returncode != 0:
            detail = (extract.stderr or extract.stdout or b"").decode("utf-8", errors="replace").strip()
            raise RunnerError(f"Failed to extract stage artifacts: {detail}")

    def _build_stage_script(
        self,
        *,
        branch_name: str,
        stage_name: str,
        trial_id: str,
        stage: RunnerStageConfig,
        metric_paths: list[str],
    ) -> tuple[str, str]:
        remote_stage_dir = f"{self.config.runtime_root.rstrip('/')}/{trial_id}/{stage_name}"
        lines = [
            "set -euo pipefail",
            f"cd {shlex.quote(self.config.repo_path)}",
        ]
        if self.config.activation_command:
            lines.append(self.config.activation_command)
        if self.git_auth is not None:
            lines.extend(self.git_auth.build_remote_fetch_setup(enabled=self.git_auth.config.use_for_remote_fetch))
        lines.extend(
            [
                f"git fetch {shlex.quote(self.config.git_remote)} {shlex.quote(branch_name)}",
                f"git checkout -B {shlex.quote(branch_name)} {shlex.quote(self.config.git_remote + '/' + branch_name)}",
                f"stage_dir={shlex.quote(remote_stage_dir)}",
                'rm -rf "$stage_dir"',
                'mkdir -p "$stage_dir/captured"',
                'printf "hostname=%s\\n" "$(hostname)" > "$stage_dir/stage.env"',
                'printf "remote_pid=%s\\n" "$$" >> "$stage_dir/stage.env"',
                f'printf "branch_name=%s\\n" {shlex.quote(branch_name)} >> "$stage_dir/stage.env"',
                'printf "started_epoch=%s\\n" "$(date +%s)" >> "$stage_dir/stage.env"',
                ': > "$stage_dir/commands.tsv"',
                'stage_status="succeeded"',
            ]
        )
        for key, value in sorted(self.config.env.items()):
            lines.append(f"export {key}={shlex.quote(str(value))}")
        for key, value in sorted(stage.env.items()):
            lines.append(f"export {key}={shlex.quote(str(value))}")
        for index, command in enumerate(stage.commands, start=1):
            stdout_name = f"command_{index:03d}.stdout"
            stderr_name = f"command_{index:03d}.stderr"
            rss_name = f"command_{index:03d}.max_rss_kb"
            quoted_command = shlex.quote(command)
            lines.extend(
                [
                    f'cmd_stdout="$stage_dir/{stdout_name}"',
                    f'cmd_stderr="$stage_dir/{stderr_name}"',
                    f'cmd_rss="$stage_dir/{rss_name}"',
                    'cmd_started="$(date +%s)"',
                    "set +e",
                    (
                        "if command -v /usr/bin/time >/dev/null 2>&1; then "
                        f"timeout {int(stage.timeout_seconds)} /usr/bin/time -f '%M' -o \"$cmd_rss\" bash -lc {quoted_command} >\"$cmd_stdout\" 2>\"$cmd_stderr\"; "
                        f"elif command -v gtime >/dev/null 2>&1; then timeout {int(stage.timeout_seconds)} gtime -f '%M' -o \"$cmd_rss\" bash -lc {quoted_command} >\"$cmd_stdout\" 2>\"$cmd_stderr\"; "
                        f"else timeout {int(stage.timeout_seconds)} bash -lc {quoted_command} >\"$cmd_stdout\" 2>\"$cmd_stderr\"; fi"
                    ),
                    "cmd_exit=$?",
                    "set -e",
                    'cmd_finished="$(date +%s)"',
                    'cmd_status="ok"',
                    'if [ "$cmd_exit" -ne 0 ]; then',
                    '  if [ "$cmd_exit" -eq 124 ]; then',
                    '    cmd_status="timeout"',
                    '    stage_status="timeout"',
                    "  else",
                    '    cmd_status="failed"',
                    '    stage_status="failed"',
                    "  fi",
                    "fi",
                    'printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
                    f'"{index}" "$cmd_exit" "$cmd_started" "$cmd_finished" "$cmd_status" "{stdout_name}" "{stderr_name}" "{rss_name}" >> "$stage_dir/commands.tsv"',
                    'if [ "$cmd_exit" -ne 0 ]; then break; fi',
                ]
            )
        for path in metric_paths:
            rel_path = str(path).strip().lstrip("./")
            if not rel_path:
                continue
            parent = str(Path(rel_path).parent)
            if parent == ".":
                rel_var = f"metric_rel_{len(lines)}"
                name_var = f"metric_name_{len(lines)}"
                lines.append(f"{rel_var}={shlex.quote(rel_path)}")
                lines.append(f"{name_var}={shlex.quote(Path(rel_path).name)}")
                lines.append(f'if [ -f "${rel_var}" ]; then cp "${rel_var}" "$stage_dir/captured/${name_var}"; fi')
            else:
                rel_var = f"metric_rel_{len(lines)}"
                dir_var = f"metric_dir_{len(lines)}"
                lines.extend(
                    [
                        f"{rel_var}={shlex.quote(rel_path)}",
                        f"{dir_var}={shlex.quote(parent)}",
                        f'if [ -f "${rel_var}" ]; then',
                        f'  mkdir -p "$stage_dir/captured/${dir_var}"',
                        f'  cp "${rel_var}" "$stage_dir/captured/${rel_var}"',
                        "fi",
                    ]
                )
        lines.extend(
            [
                'printf "status=%s\\n" "$stage_status" >> "$stage_dir/stage.env"',
                'printf "completed_epoch=%s\\n" "$(date +%s)" >> "$stage_dir/stage.env"',
            ]
        )
        return "\n".join(lines), remote_stage_dir

    def run_stage(
        self,
        *,
        trial_id: str,
        branch_name: str,
        worktree_path: Path,
        stage_name: str,
        stage: RunnerStageConfig,
        metrics: list[MetricConfig],
        artifact_dir: Path,
    ) -> StageExecution:
        del worktree_path
        metric_paths = sorted({metric.path for metric in metrics if metric.path})
        script, remote_stage_dir = self._build_stage_script(
            branch_name=branch_name,
            stage_name=stage_name,
            trial_id=trial_id,
            stage=stage,
            metric_paths=[path for path in metric_paths if path],
        )
        stage_artifact_dir = (artifact_dir / stage_name).resolve()
        stage_artifact_dir.mkdir(parents=True, exist_ok=True)
        script_timeout = max(60, len(stage.commands) * int(stage.timeout_seconds) + 120)
        try:
            proc = self._run_remote_script(script, timeout_seconds=script_timeout)
        except subprocess.TimeoutExpired as exc:
            (stage_artifact_dir / "stage_setup.stderr").write_text(str(exc), encoding="utf-8")
            now = utc_now()
            return StageExecution(
                stage_name=stage_name,
                branch_name=branch_name,
                status="timeout",
                command_results=[],
                metrics={},
                resource_metadata={"setup_timeout": True, "execution_mode": "ssh"},
                artifact_dir=stage_artifact_dir,
                started_at=now,
                completed_at=now,
            )
        (stage_artifact_dir / "stage_setup.stdout").write_text(proc.stdout or "", encoding="utf-8")
        (stage_artifact_dir / "stage_setup.stderr").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            now = utc_now()
            return StageExecution(
                stage_name=stage_name,
                branch_name=branch_name,
                status="failed",
                command_results=[],
                metrics={},
                resource_metadata={"setup_returncode": proc.returncode, "execution_mode": "ssh"},
                artifact_dir=stage_artifact_dir,
                started_at=now,
                completed_at=now,
            )
        self._fetch_stage_dir(remote_stage_dir=remote_stage_dir, local_stage_dir=stage_artifact_dir)
        stage_env = _parse_env_file(stage_artifact_dir / "stage.env")
        command_results: list[StageCommandResult] = []
        commands_tsv = stage_artifact_dir / "commands.tsv"
        if commands_tsv.exists():
            for raw_line in commands_tsv.read_text(encoding="utf-8").splitlines():
                parts = raw_line.split("\t")
                if len(parts) != 8:
                    continue
                index, exit_code, started_epoch, finished_epoch, status, stdout_name, stderr_name, rss_name = parts
                rss_path = stage_artifact_dir / rss_name
                max_rss_kb = None
                if rss_path.exists():
                    try:
                        max_rss_kb = int(rss_path.read_text(encoding="utf-8").strip())
                    except ValueError:
                        max_rss_kb = None
                command_results.append(
                    StageCommandResult(
                        index=int(index),
                        command=stage.commands[int(index) - 1],
                        exit_code=int(exit_code),
                        status=status,
                        stdout_path=(stage_artifact_dir / stdout_name),
                        stderr_path=(stage_artifact_dir / stderr_name),
                        wall_time_seconds=max(0.0, float(finished_epoch) - float(started_epoch)),
                        max_rss_kb=max_rss_kb,
                    )
                )
        extracted_metrics = extract_metrics(artifact_dir=stage_artifact_dir, metrics=metrics)
        resource_metadata = {
            "hostname": stage_env.get("hostname"),
            "remote_pid": stage_env.get("remote_pid"),
            "execution_mode": "ssh",
        }
        return StageExecution(
            stage_name=stage_name,
            branch_name=branch_name,
            status=stage_env.get("status", "failed"),
            command_results=command_results,
            metrics=extracted_metrics,
            resource_metadata=resource_metadata,
            artifact_dir=stage_artifact_dir,
            started_at=_iso_from_epoch(stage_env.get("started_epoch")),
            completed_at=_iso_from_epoch(stage_env.get("completed_epoch")),
        )


def build_trial_runner(config, *, git_auth: Optional[GitAuthManager] = None) -> BaseTrialRunner:
    if config.runner.mode == "local":
        return LocalTrialRunner(config.runner.local)
    if config.runner.mode == "ssh":
        return SSHTrialRunner(config.runner.ssh, git_auth=git_auth)
    raise ValueError(f"Unsupported runner.mode: {config.runner.mode}")
