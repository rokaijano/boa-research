from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agent_presets import resolve_agent_profile
from .init_banner import render_banner
from .init_models import InitDraft, InitSetupSelection
from .init_services import InitServices


def _comma_split(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _env_map_to_text(values: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _parse_env_map(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in _comma_split(value):
        if "=" in item:
            key, raw_value = item.split("=", 1)
            parsed[key.strip()] = raw_value.strip()
    return parsed


@dataclass
class PromptAdapter:
    select: Callable[..., str]
    text: Callable[..., str]
    confirm: Callable[..., bool]


def build_prompt_adapter() -> PromptAdapter:
    from InquirerPy import inquirer

    def select(*, message: str, choices: list[tuple[str, str]], default: str | None = None) -> str:
        return str(
            inquirer.select(
                message=message,
                choices=[{"name": label, "value": value} for label, value in choices],
                default=default,
            ).execute()
        )

    def text(*, message: str, default: str = "") -> str:
        return str(inquirer.text(message=message, default=default).execute()).strip()

    def confirm(*, message: str, default: bool = True) -> bool:
        return bool(inquirer.confirm(message=message, default=default).execute())

    return PromptAdapter(select=select, text=text, confirm=confirm)


class InitWizard:
    def __init__(
        self,
        *,
        initial_path: Path,
        services: InitServices | None = None,
        prompts: PromptAdapter | None = None,
        output: Callable[[str], None] | None = None,
    ) -> None:
        self.initial_path = initial_path
        self.services = services or InitServices()
        self.prompts = prompts or build_prompt_adapter()
        self.output = output or print
        self.draft = InitDraft()

    def _print_banner(self) -> None:
        size = shutil.get_terminal_size((120, 30))
        self.output(render_banner(width=size.columns, height=size.lines, allow_unicode=True))
        self.output("")

    def _show(self, message: str) -> None:
        self.output(message)

    def _agent_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for label, preset, binary in [
            ("Codex", "codex", "codex"),
            ("Claude Code", "claude_code", "claude"),
            ("Copilot", "copilot", "copilot"),
            ("DeepAgents", "deepagents", None),
            ("Custom command", "custom", None),
        ]:
            if binary is None:
                choices.append((label, preset))
                continue
            installed = shutil.which(binary) is not None
            suffix = "" if installed else " (not found)"
            choices.append((f"{label}{suffix}", preset))
        return choices

    def _configure_known_cli_agent(self, selection: InitSetupSelection, *, preset: str, default_command: str) -> None:
        selection.agent_command = default_command
        selection.agent_args = []
        selection.agent_env = {}
        selection.agent_profile = None
        selection.agent_model = ""
        command_missing = shutil.which(default_command) is None
        if command_missing or self.prompts.confirm(message=f"Override `{default_command}` executable?", default=False):
            selection.agent_command = self.prompts.text(message="Agent executable", default=default_command) or default_command
        if preset == "codex":
            profile_mode = self.prompts.select(
                message="Codex profile",
                choices=[("Use default profile", "default"), ("Use config profile", "profile")],
                default="default",
            )
            selection.agent_profile = (
                self.prompts.text(message="Codex profile name", default=selection.agent_profile or "") or None
            ) if profile_mode == "profile" else None
            model_mode = self.prompts.select(
                message="Codex model",
                choices=[
                    ("Use default model", "default"),
                    ("GPT-5.4", "gpt-5.4"),
                    ("GPT-5 Codex", "gpt-5-codex"),
                    ("Custom model", "custom"),
                ],
                default="default",
            )
            if model_mode == "custom":
                selection.agent_model = self.prompts.text(message="Custom Codex model", default=selection.agent_model)
            elif model_mode == "default":
                selection.agent_model = ""
            else:
                selection.agent_model = model_mode
            return
        profile_mode = self.prompts.select(
            message="CLI mode",
            choices=[("Use default mode", "default"), ("Set custom profile / mode", "custom")],
            default="default",
        )
        selection.agent_profile = (
            self.prompts.text(message="Profile / mode", default=selection.agent_profile or "") or None
        ) if profile_mode == "custom" else None
        model_mode = self.prompts.select(
            message="Model",
            choices=[("Use default model", "default"), ("Set custom model", "custom")],
            default="default",
        )
        selection.agent_model = (
            self.prompts.text(message="Custom model", default=selection.agent_model) if model_mode == "custom" else ""
        )

    def _configure_deepagents(self, selection: InitSetupSelection) -> None:
        selection.agent_command = ""
        selection.agent_args = []
        selection.agent_env = {}
        selection.agent_backend = self.prompts.select(
            message="DeepAgents backend",
            choices=[("Ollama", "ollama"), ("OpenAI", "openai")],
            default=selection.agent_backend,
        )
        if selection.agent_backend == "ollama":
            model_choice = self.prompts.select(
                message="DeepAgents model",
                choices=[
                    ("Default model", "default"),
                    ("qwen2.5-coder:14b", "qwen2.5-coder:14b"),
                    ("Custom model", "custom"),
                ],
                default="default",
            )
            selection.agent_model = (
                self.prompts.text(message="Custom Ollama model", default=selection.agent_model or "qwen2.5-coder:14b")
                if model_choice == "custom"
                else ("" if model_choice == "default" else model_choice)
            )
            base_url_mode = self.prompts.select(
                message="Ollama endpoint",
                choices=[("Use default", "default"), ("Custom base URL", "custom")],
                default="default",
            )
            selection.agent_base_url = (
                self.prompts.text(message="Ollama base URL", default=selection.agent_base_url)
                if base_url_mode == "custom"
                else "http://127.0.0.1:11434"
            )
            selection.agent_api_key_env = None
            return
        model_choice = self.prompts.select(
            message="OpenAI model",
            choices=[("Default model", "default"), ("gpt-5.4", "gpt-5.4"), ("Custom model", "custom")],
            default="default",
        )
        selection.agent_model = (
            self.prompts.text(message="Custom OpenAI model", default=selection.agent_model or "gpt-5.4")
            if model_choice == "custom"
            else ("" if model_choice == "default" else model_choice)
        )
        key_choice = self.prompts.select(
            message="API key environment variable",
            choices=[("OPENAI_API_KEY", "OPENAI_API_KEY"), ("Custom env var", "custom")],
            default=selection.agent_api_key_env or "OPENAI_API_KEY",
        )
        selection.agent_api_key_env = (
            self.prompts.text(message="Custom API key env var", default=selection.agent_api_key_env or "OPENAI_API_KEY")
            if key_choice == "custom"
            else key_choice
        )
        selection.agent_base_url = "https://api.openai.com/v1"

    def _configure_custom_agent(self, selection: InitSetupSelection) -> None:
        selection.agent_command = self.prompts.text(message="Agent command", default=selection.agent_command or "")
        args_default = " ".join(selection.agent_args)
        selection.agent_args = shlex.split(self.prompts.text(message="Agent args", default=args_default) or args_default)
        selection.agent_profile = self.prompts.text(message="Optional profile / mode", default=selection.agent_profile or "") or None
        selection.agent_model = self.prompts.text(message="Optional model", default=selection.agent_model)
        env_default = _env_map_to_text(selection.agent_env)
        selection.agent_env = _parse_env_map(self.prompts.text(message="Optional env vars (KEY=VALUE, ...)", default=env_default))

    def _select_existing_action(self) -> None:
        detected = self.draft.detected_repo
        selection = self.draft.selection
        assert detected is not None
        assert selection is not None
        existing = detected.existing_setup
        self._show(f"Detected repo root: {detected.repo_root}")
        if existing.status == "valid":
            self._show(f"\nCurrent BOA setup detected:\n{existing.summary}\n")
            action = self.prompts.select(
                message="How should BOA handle the existing setup?",
                choices=[("Review existing setup", "review"), ("Update interactively", "update"), ("Overwrite", "overwrite")],
                default=selection.existing_action,
            )
            selection.existing_action = action
            if action == "review":
                self._show("\nReview mode selected.\n")
        elif existing.status == "invalid":
            self._show("Existing BOA files are invalid or foreign. Preview:\n")
            if existing.raw_preview:
                self._show(existing.raw_preview)
            self._show("")
            selection.existing_action = "overwrite"
            self._show("BOA will overwrite these files with a fresh setup.\n")
        else:
            selection.existing_action = "overwrite"
            self._show("No BOA files detected. This is expected for first-time setup.")
            self._show("BOA will create fresh `boa.config`, `boa.md`, and `.boa/`.\n")

    def _configure_agent(self) -> None:
        selection = self.draft.selection
        assert selection is not None
        preset = self.prompts.select(
            message="Select the coding agent",
            choices=self._agent_choices(),
            default=selection.agent_preset,
        )
        defaults = resolve_agent_profile({"preset": preset})
        selection.agent_preset = preset
        selection.agent_runtime = str(defaults.get("runtime") or selection.agent_runtime)
        default_command = str(defaults.get("command") or selection.agent_command)
        if preset in {"codex", "claude_code", "copilot"}:
            self._configure_known_cli_agent(selection, preset=preset, default_command=default_command)
            return
        if selection.agent_runtime == "deepagents":
            self._configure_deepagents(selection)
            return
        self._configure_custom_agent(selection)

    def _configure_runner(self) -> None:
        selection = self.draft.selection
        assert selection is not None
        selection.runner_mode = self.prompts.select(
            message="Select the runner",
            choices=[("Local execution", "local"), ("SSH execution", "ssh")],
            default=selection.runner_mode,
        )
        if selection.runner_mode == "local":
            selection.local_activation_command = (
                self.prompts.text(
                    message="Optional local activation command",
                    default=selection.local_activation_command or "",
                )
                or None
            )
            return
        selection.ssh_host_alias = self.prompts.text(message="SSH host alias", default=selection.ssh_host_alias or "") or None
        if not selection.ssh_host_alias:
            selection.ssh_host = self.prompts.text(message="SSH host", default=selection.ssh_host or "") or None
            selection.ssh_user = self.prompts.text(message="SSH user", default=selection.ssh_user or "") or None
        port_value = self.prompts.text(message="SSH port", default=str(selection.ssh_port))
        selection.ssh_port = int(port_value or "22")
        ssh_key = self.prompts.text(
            message="Optional SSH key path",
            default=str(selection.ssh_key_path or ""),
        )
        selection.ssh_key_path = Path(ssh_key) if ssh_key else None
        selection.ssh_repo_path = self.prompts.text(message="Remote repo path", default=selection.ssh_repo_path)
        selection.ssh_git_remote = self.prompts.text(message="Remote git name", default=selection.ssh_git_remote)
        selection.ssh_activation_command = (
            self.prompts.text(
                message="Optional remote activation command",
                default=selection.ssh_activation_command or "",
            )
            or None
        )

    def _run_preflight(self) -> bool:
        assert self.draft.detected_repo is not None
        assert self.draft.selection is not None
        while True:
            checks = self.services.run_preflight(self.draft.detected_repo, self.draft.selection)
            self.draft.preflight_checks = checks
            self._show("Preflight checks:")
            for item in checks:
                status = "PASS" if item.passed else "FAIL"
                self._show(f"- {status} {item.name}: {item.detail}")
            if all(item.passed or not item.blocking for item in checks):
                self._show("")
                return True
            action = self.prompts.select(
                message="Preflight failed. What next?",
                choices=[("Retry preflight", "retry"), ("Edit answers", "edit"), ("Abort", "abort")],
                default="retry",
            )
            if action == "retry":
                continue
            if action == "edit":
                self._configure_agent()
                self._configure_runner()
                continue
            return False

    def _run_analysis(self) -> bool:
        assert self.draft.detected_repo is not None
        assert self.draft.selection is not None
        while True:
            try:
                analysis = self.services.analyze_repo(self.draft.detected_repo, self.draft.selection)
                self._show("\nRepo analysis completed.\n")
            except Exception as exc:
                self._show(f"\nAnalysis failed: {exc}\n")
                action = self.prompts.select(
                    message="How should BOA proceed?",
                    choices=[("Retry analysis", "retry"), ("Use heuristic defaults", "defaults"), ("Abort", "abort")],
                    default="defaults",
                )
                if action == "retry":
                    continue
                if action == "defaults":
                    analysis = self.services.default_repo_analysis(self.draft.detected_repo.repo_root)
                else:
                    return False
            self.draft.analysis = analysis
            self.draft.reviewed_plan = self.services.merge_reviewed_plan(self.draft.selection, analysis)
            return True

    def _review_plan(self) -> None:
        plan = self.draft.reviewed_plan
        assert plan is not None
        self._show("Proposed setup:")
        self._show(f"- Agent: {plan.selection.agent_preset}")
        self._show(f"- Runner: {plan.selection.runner_mode}")
        self._show(f"- Train command: {plan.train_command}")
        self._show(f"- Eval command: {plan.eval_command}")
        self._show(f"- Primary metric: {plan.primary_metric_name} ({plan.metric_direction})")
        self._show(f"- Editable files: {', '.join(plan.editable_files)}")
        self._show(f"- Protected files: {', '.join(plan.protected_files)}")
        if plan.selection.runner_mode == "ssh":
            self._show(
                f"- SSH target: {plan.selection.ssh_host_alias or plan.selection.ssh_host or '<unset>'}  repo={plan.selection.ssh_repo_path}"
            )
        self._show("")
        if self.prompts.confirm(message="Edit this reviewed setup before writing?", default=False):
            plan.selection.agent_preset = self.prompts.select(
                message="Agent preset",
                choices=[
                    ("Codex", "codex"),
                    ("Claude Code", "claude_code"),
                    ("Copilot", "copilot"),
                    ("DeepAgents", "deepagents"),
                    ("Custom command", "custom"),
                ],
                default=plan.selection.agent_preset,
            )
            plan.selection.runner_mode = self.prompts.select(
                message="Runner mode",
                choices=[("Local", "local"), ("SSH", "ssh")],
                default=plan.selection.runner_mode,
            )
            if plan.selection.runner_mode == "ssh":
                plan.selection.ssh_host_alias = (
                    self.prompts.text(message="SSH host alias", default=plan.selection.ssh_host_alias or "") or None
                )
                if not plan.selection.ssh_host_alias:
                    plan.selection.ssh_host = self.prompts.text(message="SSH host", default=plan.selection.ssh_host or "") or None
                plan.selection.ssh_repo_path = self.prompts.text(
                    message="SSH repo path",
                    default=plan.selection.ssh_repo_path,
                )
            plan.train_command = self.prompts.text(message="Train command", default=plan.train_command)
            plan.eval_command = self.prompts.text(message="Eval command", default=plan.eval_command)
            plan.primary_metric_name = self.prompts.text(message="Primary metric", default=plan.primary_metric_name)
            plan.metric_direction = self.prompts.select(
                message="Metric direction",
                choices=[("Maximize", "maximize"), ("Minimize", "minimize")],
                default=plan.metric_direction,
            )
            plan.editable_files = _comma_split(
                self.prompts.text(message="Editable files (comma separated)", default=", ".join(plan.editable_files))
            )
            plan.protected_files = _comma_split(
                self.prompts.text(message="Protected files (comma separated)", default=", ".join(plan.protected_files))
            )

    def _write_and_validate(self) -> None:
        plan = self.draft.reviewed_plan
        assert plan is not None
        self.draft.write_result = self.services.write_contract_files(plan)
        result = self.draft.write_result
        if result.created_paths:
            self._show("Created:")
            for path in result.created_paths:
                self._show(f"- {path}")
        if result.updated_paths:
            self._show("Updated:")
            for path in result.updated_paths:
                self._show(f"- {path}")
        if result.skipped_paths:
            self._show("Skipped:")
            for path in result.skipped_paths:
                self._show(f"- {path}")
        if self.prompts.confirm(message="Run validation now?", default=True):
            self.draft.validation = self.services.validate_written_setup(plan)
            for detail in self.draft.validation.details:
                self._show(f"- {detail}")

    def run(self) -> InitDraft:
        self._print_banner()
        detected = self.services.detect_repo(self.initial_path)
        self.draft.detected_repo = detected
        self.draft.selection = self.services.default_selection_for_repo(detected)

        self._select_existing_action()
        if self.draft.selection is not None and self.draft.selection.existing_action == "review":
            return self.draft

        self._configure_agent()
        self._configure_runner()
        if not self._run_preflight():
            return self.draft
        if not self._run_analysis():
            return self.draft
        self._review_plan()
        self._write_and_validate()
        self._show("\nNext: run `boa run {}`".format(detected.repo_root))
        return self.draft
