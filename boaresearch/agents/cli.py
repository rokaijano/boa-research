from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..prompt_builder import build_cli_execution_bundle, build_cli_planning_bundle
from ..schema import AgentExecutionContext, AgentPlanningContext, CandidateMetadata, CandidatePlan
from .base import BaseResearchAgent, ResearchAgentError, extract_json_object, parse_candidate_dict, parse_candidate_plan_dict


class CliResearchAgent(BaseResearchAgent):
    def __init__(self, *, repo_root: Path, agent_config) -> None:
        self.repo_root = repo_root
        self.agent_config = agent_config

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
        bundle = build_cli_planning_bundle(repo_root=self.repo_root, context=context)
        paths = {
            "system_prompt": bundle_root / "system_prompt.txt",
            "task_prompt": bundle_root / "task_prompt.txt",
            "prompt": bundle_root / "combined_prompt.txt",
            "plan_schema": bundle_root / "plan_schema.json",
            "plan_example": bundle_root / "plan_example.json",
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
        return paths

    def _write_execution_bundle(self, context: AgentExecutionContext) -> dict[str, Path]:
        bundle_root = context.prompt_bundle_dir
        bundle_root.mkdir(parents=True, exist_ok=True)
        bundle = build_cli_execution_bundle(repo_root=self.repo_root, context=context)
        paths = {
            "system_prompt": bundle_root / "system_prompt.txt",
            "task_prompt": bundle_root / "task_prompt.txt",
            "prompt": bundle_root / "combined_prompt.txt",
            "candidate_schema": bundle_root / "candidate_schema.json",
            "candidate_example": bundle_root / "candidate_example.json",
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
        return paths

    def _run_cli(self, *, phase: str, cwd: Path, prompt_path: Path, template_values: dict[str, str], extra_env: dict[str, str], stdout_path: Path, stderr_path: Path) -> str:
        command = self._build_command(template_values)
        env = os.environ.copy()
        env.update({key: self._render_template(value, template_values) for key, value in dict(self.agent_config.env).items()})
        env.update(extra_env)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        try:
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
        except FileNotFoundError as exc:
            raise ResearchAgentError(f"CLI agent command not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' timed out during {phase} after {self.agent_config.prepare_timeout_seconds}s"
            ) from exc
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            stdout_tail = self._tail(proc.stdout)
            stderr_tail = self._tail(proc.stderr)
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' exited with code {proc.returncode} during {phase}.\n"
                f"stdout:\n{stdout_tail or '<empty>'}\n\n"
                f"stderr:\n{stderr_tail or '<empty>'}"
            )
        return proc.stdout or ""

    def _run_codex_exec(
        self,
        *,
        phase: str,
        cwd: Path,
        prompt_path: Path,
        output_path: Path,
        schema_path: Path,
        template_values: dict[str, str],
        extra_env: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> str:
        command = [
            self._render_template(str(self.agent_config.command), template_values),
            "exec",
            "-",
            "--color",
            "never",
            "--sandbox",
            "workspace-write",
            "--ask-for-approval",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-C",
            str(cwd),
        ]
        if self.agent_config.profile:
            command.extend(["-p", str(self.agent_config.profile)])
        if self.agent_config.model:
            command.extend(["-m", str(self.agent_config.model)])
        env = os.environ.copy()
        env.update({key: self._render_template(value, template_values) for key, value in dict(self.agent_config.env).items()})
        env.update(extra_env)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        try:
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
        except FileNotFoundError as exc:
            raise ResearchAgentError(f"CLI agent command not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' timed out during {phase} after {self.agent_config.prepare_timeout_seconds}s"
            ) from exc
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            stdout_tail = self._tail(proc.stdout)
            stderr_tail = self._tail(proc.stderr)
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' exited with code {proc.returncode} during {phase}.\n"
                f"stdout:\n{stdout_tail or '<empty>'}\n\n"
                f"stderr:\n{stderr_tail or '<empty>'}"
            )
        return proc.stdout or ""

    def _parse_plan(self, *, plan_path: Path, stdout: str) -> CandidatePlan:
        if plan_path.exists():
            data = extract_json_object(plan_path.read_text(encoding="utf-8"))
            return parse_candidate_plan_dict(data)
        data = extract_json_object(stdout)
        return parse_candidate_plan_dict(data)

    def _parse_candidate(self, *, candidate_path: Path, stdout: str) -> CandidateMetadata:
        if candidate_path.exists():
            data = extract_json_object(candidate_path.read_text(encoding="utf-8"))
            return parse_candidate_dict(data)
        data = extract_json_object(stdout)
        return parse_candidate_dict(data)

    def plan_trial(self, context: AgentPlanningContext) -> CandidatePlan:
        bundle_paths = self._write_planning_bundle(context)
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
                phase="planning",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                output_path=context.plan_output_path,
                schema_path=bundle_paths["plan_schema"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        else:
            stdout = self._run_cli(
                phase="planning",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        try:
            return self._parse_plan(plan_path=context.plan_output_path, stdout=stdout)
        except ResearchAgentError as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' did not return a valid candidate plan.\n"
                f"stdout:\n{self._tail(stdout) or '<empty>'}"
            ) from exc

    def prepare_candidate(self, context: AgentExecutionContext) -> CandidateMetadata:
        bundle_paths = self._write_execution_bundle(context)
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
                phase="execution",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                output_path=context.candidate_output_path,
                schema_path=bundle_paths["candidate_schema"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        else:
            stdout = self._run_cli(
                phase="execution",
                cwd=context.worktree_path,
                prompt_path=bundle_paths["prompt"],
                template_values=template_values,
                extra_env=env,
                stdout_path=bundle_paths["stdout"],
                stderr_path=bundle_paths["stderr"],
            )
        try:
            return self._parse_candidate(candidate_path=context.candidate_output_path, stdout=stdout)
        except ResearchAgentError as exc:
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' did not return valid candidate metadata.\n"
                f"stdout:\n{self._tail(stdout) or '<empty>'}"
            ) from exc
