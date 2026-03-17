from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..prompt_builder import build_cli_prompt_bundle
from ..schema import CandidateMetadata
from .base import BaseResearchAgent, ResearchAgentError, extract_json_object, parse_candidate_dict


class CliResearchAgent(BaseResearchAgent):
    def __init__(self, *, repo_root: Path, agent_config) -> None:
        self.repo_root = repo_root
        self.agent_config = agent_config

    def _write_prompt_bundle(self, context) -> dict[str, Path]:
        bundle_root = context.prompt_bundle_dir
        bundle_root.mkdir(parents=True, exist_ok=True)
        bundle = build_cli_prompt_bundle(repo_root=self.repo_root, context=context)
        paths = {
            "system_prompt": bundle_root / "system_prompt.txt",
            "task_prompt": bundle_root / "task_prompt.txt",
            "prompt": bundle_root / "combined_prompt.txt",
            "candidate_schema": bundle_root / "candidate_schema.json",
            "candidate_example": bundle_root / "candidate_example.json",
            "candidate": bundle_root / "candidate.json",
            "stdout": bundle_root / "last_stdout.txt",
            "stderr": bundle_root / "last_stderr.txt",
        }
        paths["system_prompt"].write_text(bundle["system_prompt"], encoding="utf-8")
        paths["task_prompt"].write_text(bundle["task_prompt"], encoding="utf-8")
        paths["prompt"].write_text(bundle["combined_prompt"], encoding="utf-8")
        paths["candidate_schema"].write_text(bundle["candidate_schema"], encoding="utf-8")
        paths["candidate_example"].write_text(bundle["candidate_example"], encoding="utf-8")
        if paths["candidate"].exists():
            paths["candidate"].unlink()
        return paths

    def _template_values(self, context, bundle_paths: dict[str, Path]) -> dict[str, str]:
        return {
            "agent": str(self.agent_config.preset),
            "model": str(self.agent_config.model),
            "reasoning_effort": str(self.agent_config.reasoning_effort or ""),
            "max_agent_steps": str(context.max_agent_steps),
            "run_tag": context.run_tag,
            "trial_id": context.trial_id,
            "worktree_path": str(context.worktree_path),
            "prompt_path": str(bundle_paths["prompt"]),
            "system_prompt_path": str(bundle_paths["system_prompt"]),
            "task_prompt_path": str(bundle_paths["task_prompt"]),
            "candidate_schema_path": str(bundle_paths["candidate_schema"]),
            "candidate_example_path": str(bundle_paths["candidate_example"]),
            "candidate_path": str(bundle_paths["candidate"]),
            "boa_md_path": str(context.boa_md_path),
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

    @staticmethod
    def _tail(text: str, *, limit: int = 4000) -> str:
        cleaned = str(text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[-limit:]

    def _parse_candidate(self, *, candidate_path: Path, stdout: str) -> CandidateMetadata:
        if candidate_path.exists():
            data = extract_json_object(candidate_path.read_text(encoding="utf-8"))
            return parse_candidate_dict(data)
        data = extract_json_object(stdout)
        return parse_candidate_dict(data)

    def prepare_candidate(self, context) -> CandidateMetadata:
        bundle_paths = self._write_prompt_bundle(context)
        template_values = self._template_values(context, bundle_paths)
        command = self._build_command(template_values)
        env = os.environ.copy()
        env.update({key: self._render_template(value, template_values) for key, value in dict(self.agent_config.env).items()})
        env.update(
            {
                "BOA_AGENT": str(self.agent_config.preset),
                "BOA_RUN_TAG": context.run_tag,
                "BOA_TRIAL_ID": context.trial_id,
                "BOA_WORKTREE": str(context.worktree_path),
                "BOA_PROMPT_PATH": str(bundle_paths["prompt"]),
                "BOA_SYSTEM_PROMPT_PATH": str(bundle_paths["system_prompt"]),
                "BOA_TASK_PROMPT_PATH": str(bundle_paths["task_prompt"]),
                "BOA_CANDIDATE_SCHEMA_PATH": str(bundle_paths["candidate_schema"]),
                "BOA_CANDIDATE_EXAMPLE_PATH": str(bundle_paths["candidate_example"]),
                "BOA_CANDIDATE_METADATA_PATH": str(bundle_paths["candidate"]),
                "BOA_MD_PATH": str(context.boa_md_path),
            }
        )
        prompt_text = bundle_paths["prompt"].read_text(encoding="utf-8")
        try:
            proc = subprocess.run(
                command,
                cwd=str(context.worktree_path),
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
                f"CLI agent '{self.agent_config.preset}' timed out after {self.agent_config.prepare_timeout_seconds}s"
            ) from exc
        bundle_paths["stdout"].write_text(proc.stdout or "", encoding="utf-8")
        bundle_paths["stderr"].write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            stdout_tail = self._tail(proc.stdout)
            stderr_tail = self._tail(proc.stderr)
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' exited with code {proc.returncode}.\n"
                f"stdout:\n{stdout_tail or '<empty>'}\n\n"
                f"stderr:\n{stderr_tail or '<empty>'}"
            )
        try:
            return self._parse_candidate(candidate_path=bundle_paths["candidate"], stdout=proc.stdout or "")
        except ResearchAgentError as exc:
            stdout_tail = self._tail(proc.stdout)
            raise ResearchAgentError(
                f"CLI agent '{self.agent_config.preset}' did not return valid candidate metadata.\n"
                f"stdout:\n{stdout_tail or '<empty>'}"
            ) from exc
