from __future__ import annotations

import os
import shutil
import sys
import subprocess
from pathlib import Path

from ..process import run_process_with_live_output
from ..prompt_builder import build_cli_execution_bundle, build_cli_planning_bundle
from ..runtime.observer import RunEvent
from ..schema import AgentExecutionContext, AgentPlanningContext, CandidateMetadata, CandidatePlan
from .base import BaseResearchAgent, ResearchAgentError
from .interaction import BoaInteractionLayer


class CliResearchAgent(BaseResearchAgent):
    def __init__(self, *, repo_root: Path, agent_config, observer=None) -> None:
        self.repo_root = repo_root
        self.agent_config = agent_config
        self.observer = observer
        self.interaction = BoaInteractionLayer()

    def _emit(
        self,
        *,
        kind: str,
        message: str,
        trial_id: str,
        phase: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.observer is None:
            return
        self.observer.emit(
            RunEvent(
                kind=kind,
                message=message,
                trial_id=trial_id,
                phase=phase,
                metadata=dict(metadata or {}),
            )
        )

    def _template_values(
        self,
        *,
        phase: str,
        trial_id: str,
        worktree_path: Path,
        prompt_dir: Path,
        tool_context_path: Path,
        plan_path: Path,
        candidate_path: Path,
    ) -> dict[str, str]:
        return {
            "agent": str(self.agent_config.preset),
            "model": str(self.agent_config.model),
            "reasoning_effort": str(self.agent_config.reasoning_effort or ""),
            "phase": phase,
            "trial_id": str(trial_id),
            "worktree_path": str(worktree_path),
            "prompt_dir": str(prompt_dir),
            "prompt_path": str(prompt_dir / "combined_prompt.txt"),
            "system_prompt_path": str(prompt_dir / "system_prompt.txt"),
            "task_prompt_path": str(prompt_dir / "task_prompt.txt"),
            "tool_context_path": str(tool_context_path),
            "plan_path": str(plan_path),
            "candidate_path": str(candidate_path),
            "candidate_metadata_path": str(candidate_path),
        }

    def _render_template(self, template: str, values: dict[str, str]) -> str:
        try:
            return str(template).format_map(values)
        except KeyError as exc:
            raise ResearchAgentError(f"Unknown CLI agent template key: {exc.args[0]}") from exc

    def _build_command(self, values: dict[str, str]) -> list[str]:
        if not self.agent_config.command:
            raise ResearchAgentError("CLI agent is missing agent.command")
        command = self._render_template(str(self.agent_config.command), values)
        args = [self._render_template(arg, values) for arg in list(self.agent_config.args)]
        return [command, *args]

    def _append_model_arg_if_supported(self, command: list[str]) -> list[str]:
        model_name = str(self.agent_config.model or "").strip()
        if not model_name:
            return command
        existing = [str(arg).strip().lower() for arg in command[1:]]
        if "-m" in existing or "--model" in existing:
            return command
        preset = str(self.agent_config.preset or "").strip().lower()
        if preset == "copilot":
            return [*command, "--model", model_name]
        if preset in {"codex", "claude_code"}:
            return [*command, "-m", model_name]
        return command

    def _append_noninteractive_flags_if_supported(self, command: list[str]) -> list[str]:
        preset = str(self.agent_config.preset or "").strip().lower()
        if preset != "copilot":
            return command
        existing = {str(arg).strip().lower() for arg in command[1:]}
        updated = list(command)
        if "--allow-all-tools" not in existing:
            updated.append("--allow-all-tools")
        return updated

    @staticmethod
    def _resolve_command_path(command: str) -> str:
        executable = str(command or "").strip()
        if not executable:
            return executable
        resolved = shutil.which(executable)
        if resolved:
            return resolved
        return executable

    @classmethod
    def _resolve_command_argv(cls, command: list[str]) -> list[str]:
        if not command:
            return command
        executable = cls._resolve_command_path(command[0])
        if os.name == "nt" and executable.lower().endswith(".ps1"):
            return [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                executable,
                *command[1:],
            ]
        return [executable, *command[1:]]

    def _is_codex_exec_adapter(self, values: dict[str, str]) -> bool:
        if str(self.agent_config.preset).strip().lower() != "codex":
            return False
        if not self.agent_config.command:
            return False
        command = self._render_template(str(self.agent_config.command), values)
        return Path(command).name == "codex"

    @staticmethod
    def _tail(text: str, *, limit: int = 4000) -> str:
        cleaned = str(text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[-limit:]

    def _write_planning_bundle(self, context: AgentPlanningContext) -> dict[str, Path]:
        bundle_root = context.prompt_bundle_dir
        bundle_root.mkdir(parents=True, exist_ok=True)
        tool_command = self._write_tool_launcher(bundle_root=bundle_root, tool_context_path=context.tool_context_path)
        bundle = build_cli_planning_bundle(repo_root=self.repo_root, context=context, tool_command=str(tool_command))
        paths = {
            "system_prompt": bundle_root / "system_prompt.txt",
            "task_prompt": bundle_root / "task_prompt.txt",
            "prompt": bundle_root / "combined_prompt.txt",
            "plan_schema": bundle_root / "plan_schema.json",
            "plan_example": bundle_root / "plan_example.json",
            "last_message": bundle_root / "last_message.txt",
            "tool_launcher": tool_command,
            "stdout": bundle_root / "last_stdout.txt",
            "stderr": bundle_root / "last_stderr.txt",
        }
        paths["system_prompt"].write_text(bundle["system_prompt"], encoding="utf-8")
        paths["task_prompt"].write_text(bundle["task_prompt"], encoding="utf-8")
        paths["prompt"].write_text(bundle["combined_prompt"], encoding="utf-8")
        paths["plan_schema"].write_text(bundle["plan_schema"], encoding="utf-8")
        paths["plan_example"].write_text(bundle["plan_example"], encoding="utf-8")
        if context.plan_output_path.exists():
            context.plan_output_path.unlink()
        if paths["last_message"].exists():
            paths["last_message"].unlink()
        return paths

    def _write_execution_bundle(self, context: AgentExecutionContext) -> dict[str, Path]:
        bundle_root = context.prompt_bundle_dir
        bundle_root.mkdir(parents=True, exist_ok=True)
        tool_command = self._write_tool_launcher(bundle_root=bundle_root, tool_context_path=context.tool_context_path)
        bundle = build_cli_execution_bundle(repo_root=self.repo_root, context=context, tool_command=str(tool_command))
        paths = {
            "system_prompt": bundle_root / "system_prompt.txt",
            "task_prompt": bundle_root / "task_prompt.txt",
            "prompt": bundle_root / "combined_prompt.txt",
            "candidate_schema": bundle_root / "candidate_schema.json",
            "candidate_example": bundle_root / "candidate_example.json",
            "last_message": bundle_root / "last_message.txt",
            "tool_launcher": tool_command,
            "stdout": bundle_root / "last_stdout.txt",
            "stderr": bundle_root / "last_stderr.txt",
        }
        paths["system_prompt"].write_text(bundle["system_prompt"], encoding="utf-8")
        paths["task_prompt"].write_text(bundle["task_prompt"], encoding="utf-8")
        paths["prompt"].write_text(bundle["combined_prompt"], encoding="utf-8")
        paths["candidate_schema"].write_text(bundle["candidate_schema"], encoding="utf-8")
        paths["candidate_example"].write_text(bundle["candidate_example"], encoding="utf-8")
        if context.candidate_output_path.exists():
            context.candidate_output_path.unlink()
        if paths["last_message"].exists():
            paths["last_message"].unlink()
        return paths

    def _write_tool_launcher(self, *, bundle_root: Path, tool_context_path: Path) -> Path:
        launcher_name = "boa-tools.cmd" if os.name == "nt" else "boa-tools"
        launcher_path = bundle_root / launcher_name
        pythonpath_entries = [str(self.repo_root)]
        existing_pythonpath = os.environ.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        if os.name == "nt":
            launcher_path.write_text(
                "\r\n".join(
                    [
                        "@echo off",
                        "setlocal",
                        f'set "BOA_TOOL_CONTEXT_PATH={tool_context_path}"',
                        f'set "PYTHONPATH={";".join(pythonpath_entries)}"',
                        f'"{sys.executable}" -m boaresearch.cli %*',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        else:
            launcher_path.write_text(
                "\n".join(
                    [
                        "#!/bin/sh",
                        "set -eu",
                        f'export BOA_TOOL_CONTEXT_PATH="{tool_context_path}"',
                        f'export PYTHONPATH="{":".join(pythonpath_entries)}"',
                        f'exec "{sys.executable}" -m boaresearch.cli "$@"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            launcher_path.chmod(0o755)
        return launcher_path

    def _run_cli(
        self,
        *,
        trial_id: str,
        phase: str,
        cwd: Path,
        prompt_path: Path,
        template_values: dict[str, str],
        extra_env: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> str:
        command = self._resolve_command_argv(
            self._append_noninteractive_flags_if_supported(
                self._append_model_arg_if_supported(self._build_command(template_values))
            )
        )
        env = os.environ.copy()
        env.update({key: self._render_template(value, template_values) for key, value in dict(self.agent_config.env).items()})
        env.update(extra_env)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        self._emit(
            kind="agent_prompt_sent",
            message=f"Sending {phase} prompt to {self.agent_config.preset}",
            trial_id=trial_id,
            phase=phase,
        )
        self._emit(
            kind="agent_command_started",
            message=f"Waiting for {self.agent_config.preset} {phase} response",
            trial_id=trial_id,
            phase=phase,
            metadata={"command": command},
        )
        try:
            if self.observer is None:
                proc = subprocess.run(
                    command,
                    cwd=str(cwd),
                    input=prompt_text,
                    capture_output=True,
                    text=True,
                    timeout=int(self.agent_config.prepare_timeout_seconds),
                    check=False,
                    env=env,
                )
            else:
                proc = run_process_with_live_output(
                    command,
                    cwd=cwd,
                    env=env,
                    input_text=prompt_text,
                    timeout_seconds=int(self.agent_config.prepare_timeout_seconds),
                    observer=self.observer,
                    trial_id=trial_id,
                    phase=phase,
                    stage_name=None,
                    stdout_source="agent.stdout",
                    stderr_source="agent.stderr",
                )
        except FileNotFoundError as exc:
            raise ResearchAgentError(f"CLI agent command not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' timed out during {phase} after {self.agent_config.prepare_timeout_seconds}s"
            ) from exc
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        self._emit(
            kind="agent_command_completed",
            message=f"{self.agent_config.preset} {phase} finished with exit code {proc.returncode}",
            trial_id=trial_id,
            phase=phase,
            metadata={"returncode": proc.returncode},
        )
        return proc.stdout or ""

    def _run_codex_exec(
        self,
        *,
        trial_id: str,
        phase: str,
        cwd: Path,
        prompt_path: Path,
        last_message_path: Path,
        template_values: dict[str, str],
        extra_env: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> str:
        command = self._resolve_command_argv(
            [
                self._render_template(str(self.agent_config.command), template_values),
                "exec",
                "-",
                "--color",
                "never",
                "--sandbox",
                "workspace-write",
                "--output-last-message",
                str(last_message_path),
                "-C",
                str(cwd),
            ]
        )
        if self.agent_config.profile:
            command.extend(["-p", str(self.agent_config.profile)])
        if self.agent_config.model:
            command.extend(["-m", str(self.agent_config.model)])
        env = os.environ.copy()
        env.update({key: self._render_template(value, template_values) for key, value in dict(self.agent_config.env).items()})
        env.update(extra_env)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        self._emit(
            kind="agent_prompt_sent",
            message=f"Sending {phase} prompt to {self.agent_config.preset}",
            trial_id=trial_id,
            phase=phase,
        )
        self._emit(
            kind="agent_command_started",
            message=f"Waiting for {self.agent_config.preset} {phase} response",
            trial_id=trial_id,
            phase=phase,
            metadata={"command": command},
        )
        try:
            if self.observer is None:
                proc = subprocess.run(
                    command,
                    cwd=str(cwd),
                    input=prompt_text,
                    capture_output=True,
                    text=True,
                    timeout=int(self.agent_config.prepare_timeout_seconds),
                    check=False,
                    env=env,
                )
            else:
                proc = run_process_with_live_output(
                    command,
                    cwd=cwd,
                    env=env,
                    input_text=prompt_text,
                    timeout_seconds=int(self.agent_config.prepare_timeout_seconds),
                    observer=self.observer,
                    trial_id=trial_id,
                    phase=phase,
                    stage_name=None,
                    stdout_source="agent.stdout",
                    stderr_source="agent.stderr",
                )
        except FileNotFoundError as exc:
            raise ResearchAgentError(f"CLI agent command not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' timed out during {phase} after {self.agent_config.prepare_timeout_seconds}s"
            ) from exc
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        last_message = ""
        if last_message_path.exists():
            last_message = last_message_path.read_text(encoding="utf-8")
        self._emit(
            kind="agent_command_completed",
            message=f"{self.agent_config.preset} {phase} finished with exit code {proc.returncode}",
            trial_id=trial_id,
            phase=phase,
            metadata={"returncode": proc.returncode},
        )
        return last_message or proc.stdout or ""

    def plan_trial(self, context: AgentPlanningContext) -> CandidatePlan:
        bundle_paths = self._write_planning_bundle(context)
        self._emit(
            kind="planning_bundle_ready",
            message="Planning prompt prepared: inspect accepted workspace, choose a BOA parent, and explain the proposed patch.",
            trial_id=context.trial_id,
            phase="planning",
            metadata={"prompt_dir": str(context.prompt_bundle_dir)},
        )
        template_values = self._template_values(
            phase="planning",
            trial_id=context.trial_id,
            worktree_path=context.worktree_path,
            prompt_dir=context.prompt_bundle_dir,
            tool_context_path=context.tool_context_path,
            plan_path=context.plan_output_path,
            candidate_path=context.plan_output_path,
        )
        env = {
            "BOA_AGENT": str(self.agent_config.preset),
            "BOA_AGENT_PHASE": "planning",
            "BOA_RUN_TAG": context.run_tag,
            "BOA_TRIAL_ID": context.trial_id,
            "BOA_WORKTREE": str(context.worktree_path),
            "BOA_PROMPT_PATH": str(bundle_paths["prompt"]),
            "BOA_SYSTEM_PROMPT_PATH": str(bundle_paths["system_prompt"]),
            "BOA_TASK_PROMPT_PATH": str(bundle_paths["task_prompt"]),
            "BOA_PLAN_SCHEMA_PATH": str(bundle_paths["plan_schema"]),
            "BOA_PLAN_EXAMPLE_PATH": str(bundle_paths["plan_example"]),
            "BOA_PLAN_PATH": str(context.plan_output_path),
            "BOA_CANDIDATE_PLAN_PATH": str(context.plan_output_path),
            "BOA_TOOL_CONTEXT_PATH": str(context.tool_context_path),
            "BOA_MD_PATH": str(context.boa_md_path),
        }
        if self._is_codex_exec_adapter(template_values):
            stdout = self._run_codex_exec(
                trial_id=context.trial_id,
                phase="planning",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                last_message_path=bundle_paths["last_message"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        else:
            stdout = self._run_cli(
                trial_id=context.trial_id,
                phase="planning",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        try:
            plan = self.interaction.parse_plan_output(plan_path=context.plan_output_path, stdout=stdout)
        except ResearchAgentError as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' did not return a valid candidate plan.\n"
                f"stdout:\n{self._tail(stdout) or '<empty>'}"
            ) from exc
        self.interaction.persist_plan(plan_path=context.plan_output_path, plan=plan)
        summary_parts = [
            f"Plan ready: {plan.patch_category}/{plan.operation_type}",
            f"parent={plan.selected_parent_branch}",
            f"risk={plan.estimated_risk:.2f}",
        ]
        if plan.target_symbols:
            summary_parts.append(f"targets={', '.join(plan.target_symbols[:3])}")
        self._emit(
            kind="planning_result",
            message=" | ".join(summary_parts),
            trial_id=context.trial_id,
            phase="planning",
        )
        return plan

    def prepare_candidate(self, context: AgentExecutionContext) -> CandidateMetadata:
        bundle_paths = self._write_execution_bundle(context)
        self._emit(
            kind="execution_bundle_ready",
            message="Execution prompt prepared: apply the patch in the trial worktree, obey guardrails, and emit candidate metadata.",
            trial_id=context.trial_id,
            phase="execution",
            metadata={"prompt_dir": str(context.prompt_bundle_dir)},
        )
        template_values = self._template_values(
            phase="execution",
            trial_id=context.trial_id,
            worktree_path=context.worktree_path,
            prompt_dir=context.prompt_bundle_dir,
            tool_context_path=context.tool_context_path,
            plan_path=context.plan_output_path,
            candidate_path=context.candidate_output_path,
        )
        env = {
            "BOA_AGENT": str(self.agent_config.preset),
            "BOA_AGENT_PHASE": "execution",
            "BOA_RUN_TAG": context.run_tag,
            "BOA_TRIAL_ID": context.trial_id,
            "BOA_WORKTREE": str(context.worktree_path),
            "BOA_PROMPT_PATH": str(bundle_paths["prompt"]),
            "BOA_SYSTEM_PROMPT_PATH": str(bundle_paths["system_prompt"]),
            "BOA_TASK_PROMPT_PATH": str(bundle_paths["task_prompt"]),
            "BOA_CANDIDATE_SCHEMA_PATH": str(bundle_paths["candidate_schema"]),
            "BOA_CANDIDATE_EXAMPLE_PATH": str(bundle_paths["candidate_example"]),
            "BOA_CANDIDATE_METADATA_PATH": str(context.candidate_output_path),
            "BOA_CANDIDATE_PATH": str(context.candidate_output_path),
            "BOA_CANDIDATE_PLAN_PATH": str(context.plan_output_path),
            "BOA_TOOL_CONTEXT_PATH": str(context.tool_context_path),
            "BOA_MD_PATH": str(context.boa_md_path),
        }
        if self._is_codex_exec_adapter(template_values):
            stdout = self._run_codex_exec(
                trial_id=context.trial_id,
                phase="execution",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                last_message_path=bundle_paths["last_message"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        else:
            stdout = self._run_cli(
                trial_id=context.trial_id,
                phase="execution",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        try:
            candidate = self.interaction.parse_candidate_output(candidate_path=context.candidate_output_path, stdout=stdout)
        except ResearchAgentError as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' did not return valid candidate metadata.\n"
                f"stdout:\n{self._tail(stdout) or '<empty>'}"
            ) from exc
        self.interaction.persist_candidate(candidate_path=context.candidate_output_path, candidate=candidate)
        summary_parts = [
            f"Candidate ready: {candidate.patch_category}/{candidate.operation_type}",
            f"risk={candidate.estimated_risk:.2f}",
        ]
        if candidate.numeric_knobs:
            first_knob = next(iter(candidate.numeric_knobs.items()))
            summary_parts.append(f"knob={first_knob[0]}={first_knob[1]}")
        self._emit(
            kind="execution_result",
            message=" | ".join(summary_parts),
            trial_id=context.trial_id,
            phase="execution",
        )
        return candidate
