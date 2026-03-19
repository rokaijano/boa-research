from __future__ import annotations

import shlex
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..agent_presets import ModelDiscoveryResult, check_model_availability_for_agent, discover_available_models_for_agent, resolve_agent_profile
from .banner import render_banner
from .models import InitDraft, InitSetupSelection
from .services import InitServices


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

    select_prompt = getattr(inquirer, "select")
    text_prompt = getattr(inquirer, "text")
    confirm_prompt = getattr(inquirer, "confirm")

    def select(*, message: str, choices: list[tuple[str, str]], default: str | None = None) -> str:
        return str(
            select_prompt(
                message=message,
                choices=[{"name": label, "value": value} for label, value in choices],
                default=default,
            ).execute()
        )

    def text(*, message: str, default: str = "") -> str:
        return str(text_prompt(message=message, default=default).execute()).strip()

    def confirm(*, message: str, default: bool = True) -> bool:
        return bool(confirm_prompt(message=message, default=default).execute())

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

    def _run_with_ephemeral_status(self, message: str, action: Callable[[], Any]) -> Any:
        if self.output is not print:
            self._show(message)
            return action()

        stop_event = threading.Event()
        status_prefix = "\r\x1b[2K"
        frames = ["■□□", "□■□", "□□■", "□■□"]

        def render() -> None:
            frame_index = 0
            while not stop_event.wait(0.12):
                frame = frames[frame_index % len(frames)]
                sys.stdout.write(f"{status_prefix}{frame} {message}")
                sys.stdout.flush()
                frame_index += 1

        worker = threading.Thread(target=render, daemon=True)
        worker.start()
        try:
            time.sleep(0.02)
            return action()
        finally:
            stop_event.set()
            worker.join(timeout=0.5)
            sys.stdout.write(f"{status_prefix}\r")
            sys.stdout.flush()

    def _show_validated_model(self, *, prompt_message: str, selected_model: str, default_label: str) -> None:
        model_label = selected_model or default_label
        rendered_line = f"? {prompt_message} {model_label} ✓"
        if self.output is not print or not getattr(sys.stdout, "isatty", lambda: False)():
            self._show(rendered_line)
            return
        sys.stdout.write(f"\x1b[1A\r\x1b[2K{rendered_line}\n")
        sys.stdout.flush()

    @staticmethod
    def _agent_display_name(preset: str) -> str:
        labels = {
            "codex": "Codex",
            "claude_code": "Claude Code",
            "copilot": "Copilot",
            "deepagents": "DeepAgents",
            "custom": "custom agent",
        }
        return labels.get(str(preset or "").strip().lower(), "agent")

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

    def _default_model_choice(self, *, current_model: str, choices: list[tuple[str, str]]) -> str:
        values = {value for _, value in choices}
        if current_model and current_model in values:
            return current_model
        if current_model:
            return "custom"
        return "default"

    def _show_model_discovery(self, *, label: str, discovery: ModelDiscoveryResult) -> None:
        if discovery.models:
            source = f" via {discovery.source}" if discovery.source else ""
            self._show(f"{label}: loaded {len(discovery.models)} model(s){source}.")
            if discovery.warning:
                self._show(f"{label}: {discovery.warning}")
            return
        if discovery.warning:
            self._show(f"{label}: {discovery.warning}")
            return
        if discovery.supports_listing:
            self._show(f"{label}: no models were discovered. You can still enter a model manually.")

    @staticmethod
    def _mark_unavailable_models(
        choices: list[tuple[str, str]],
        *,
        unavailable_models: set[str],
    ) -> list[tuple[str, str]]:
        updated_choices: list[tuple[str, str]] = []
        for label, value in choices:
            if value in unavailable_models and value not in {"default", "custom"}:
                updated_choices.append((f"{label} (not available)", value))
            else:
                updated_choices.append((label, value))
        return updated_choices

    @staticmethod
    def _model_prompt_choices(discovery: ModelDiscoveryResult, *, default_label: str = "Use default model") -> list[tuple[str, str]]:
        seen: set[str] = set()
        deduped_models: list[tuple[str, str]] = []
        for label, value in discovery.models:
            if value in seen:
                continue
            seen.add(value)
            deduped_models.append((label, value))
        return [(default_label, "default"), *deduped_models, ("Custom model", "custom")]

    def _prompt_for_model(
        self,
        *,
        message: str,
        current_model: str,
        choices: list[tuple[str, str]],
        custom_message: str,
        custom_default: str = "",
    ) -> str:
        default_choice = self._default_model_choice(current_model=current_model, choices=choices)
        model_choice = self.prompts.select(
            message=message,
            choices=choices,
            default=default_choice,
        )
        if model_choice == "custom":
            return self.prompts.text(message=custom_message, default=current_model or custom_default)
        if model_choice == "default":
            return ""
        return model_choice

    def _prompt_for_discovered_model(
        self,
        *,
        label: str,
        message: str,
        current_model: str,
        discovery: ModelDiscoveryResult,
        custom_message: str,
        custom_default: str = "",
        default_label: str = "Use default model",
        availability_kwargs: dict[str, Any] | None = None,
    ) -> str:
        self._show_model_discovery(label=label, discovery=discovery)
        unavailable_models: set[str] = set()
        base_choices = self._model_prompt_choices(discovery, default_label=default_label)
        choice_values = {value for _, value in base_choices}
        selected_model = current_model
        validation_options = dict(availability_kwargs or {})
        while True:
            selected_model = self._prompt_for_model(
                message=message,
                current_model=selected_model,
                choices=self._mark_unavailable_models(base_choices, unavailable_models=unavailable_models),
                custom_message=custom_message,
                custom_default=custom_default,
            )
            availability = self._run_with_ephemeral_status(
                "Checking model availability",
                lambda: check_model_availability_for_agent(
                    model=selected_model,
                    **validation_options,
                ),
            )
            if availability.status != "unavailable":
                if availability.status == "available":
                    self._show_validated_model(
                        prompt_message=message,
                        selected_model=selected_model,
                        default_label=default_label,
                    )
                if availability.status == "unknown" and availability.message:
                    self._show(f"Model availability check: {availability.message}")
                return selected_model
            unavailable_models.add(selected_model)
            if availability.message:
                self._show(f"Model availability check: {availability.message}")
            if selected_model in choice_values:
                selected_model = ""

    def _configure_known_cli_agent(self, selection: InitSetupSelection, *, preset: str, default_command: str) -> None:
        selection.agent_command = default_command
        selection.agent_args = []
        selection.agent_env = {}
        selection.agent_profile = None
        selection.agent_model = ""
        if shutil.which(default_command) is None:
            agent_name = self._agent_display_name(preset)
            selection.agent_command = (
                self.prompts.text(message=f"{agent_name} command or path", default=default_command) or default_command
            )
        if preset == "codex":
            discovery = discover_available_models_for_agent(
                preset=preset,
                command=selection.agent_command,
            )
            selection.agent_model = self._prompt_for_discovered_model(
                label="Codex model discovery",
                message="Codex model",
                current_model=selection.agent_model,
                discovery=discovery,
                custom_message="Custom Codex model",
                availability_kwargs={
                    "preset": preset,
                    "command": selection.agent_command,
                    "api_key_env": "OPENAI_API_KEY",
                },
            )
            return
        discovery = discover_available_models_for_agent(preset=preset)
        selection.agent_model = self._prompt_for_discovered_model(
            label="Model discovery",
            message="Model",
            current_model=selection.agent_model,
            discovery=discovery,
            custom_message="Custom model",
            availability_kwargs={
                "preset": preset,
                "command": selection.agent_command,
                "api_key_env": "ANTHROPIC_API_KEY" if preset == "claude_code" else None,
            },
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
            discovery = discover_available_models_for_agent(
                preset="deepagents",
                backend=selection.agent_backend,
                base_url=selection.agent_base_url,
            )
            selection.agent_model = self._prompt_for_discovered_model(
                label="DeepAgents model discovery",
                message="DeepAgents model",
                current_model=selection.agent_model,
                discovery=discovery,
                custom_message="Custom Ollama model",
                custom_default="qwen2.5-coder:14b",
                default_label="Default model",
                availability_kwargs={
                    "preset": "deepagents",
                    "backend": selection.agent_backend,
                    "base_url": selection.agent_base_url,
                },
            )
            selection.agent_api_key_env = None
            return
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
        base_url_mode = self.prompts.select(
            message="OpenAI-compatible endpoint",
            choices=[("Use default", "default"), ("Custom base URL", "custom")],
            default="default",
        )
        selection.agent_base_url = (
            self.prompts.text(message="OpenAI-compatible base URL", default=selection.agent_base_url)
            if base_url_mode == "custom"
            else "https://api.openai.com/v1"
        )
        discovery = discover_available_models_for_agent(
            preset="deepagents",
            backend=selection.agent_backend,
            api_key_env=selection.agent_api_key_env,
            base_url=selection.agent_base_url,
        )
        selection.agent_model = self._prompt_for_discovered_model(
            label="DeepAgents model discovery",
            message="OpenAI model",
            current_model=selection.agent_model,
            discovery=discovery,
            custom_message="Custom OpenAI model",
            custom_default="gpt-5.4",
            default_label="Default model",
            availability_kwargs={
                "preset": "deepagents",
                "backend": selection.agent_backend,
                "base_url": selection.agent_base_url,
                "api_key_env": selection.agent_api_key_env,
            },
        )

    def _configure_custom_agent(self, selection: InitSetupSelection) -> None:
        selection.agent_command = self.prompts.text(message="Agent command", default=selection.agent_command or "")
        args_default = " ".join(selection.agent_args)
        selection.agent_args = shlex.split(self.prompts.text(message="Agent args", default=args_default) or args_default)
        selection.agent_profile = self.prompts.text(message="Optional profile / mode", default=selection.agent_profile or "") or None
        selection.agent_model = self.prompts.text(message="Optional model", default=selection.agent_model)
        env_default = _env_map_to_text(selection.agent_env)
        selection.agent_env = _parse_env_map(self.prompts.text(message="Optional env vars (KEY=VALUE, ...)", default=env_default))

    def _configure_agent_aggressiveness(self) -> None:
        selection = self.draft.selection
        assert selection is not None
        selection.agent_aggressiveness = self.prompts.select(
            message="Coding aggressiveness for boa.md",
            choices=[
                ("Light", "light"),
                ("Normal", "normal"),
                ("Aggressive", "aggressive"),
            ],
            default=selection.agent_aggressiveness,
        )

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
        elif selection.agent_runtime == "deepagents":
            self._configure_deepagents(selection)
        else:
            self._configure_custom_agent(selection)
        self._configure_agent_aggressiveness()

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
        detected_repo = self.draft.detected_repo
        selection = self.draft.selection
        while True:
            try:
                analysis = self._run_with_ephemeral_status(
                    "Analyzing repo",
                    lambda: self.services.analyze_repo(detected_repo, selection),
                )
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
        config = self.services.build_config_from_plan(plan)
        self._show("Proposed setup:")
        self._show(f"- Agent: {plan.selection.agent_preset}")
        self._show(f"- Coding aggressiveness: {plan.selection.agent_aggressiveness}")
        self._show(f"- Runner: {plan.selection.runner_mode}")
        self._show(f"- Primary metric: {plan.primary_metric_name} ({plan.metric_direction})")
        self._show(f"- Max trials per run: {config.run.max_trials}")
        self._show(f"- Max consecutive failures: {config.run.max_consecutive_failures}")
        self._show(f"- Scout stage timeout: {config.runner.scout.timeout_seconds}s")
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
            plan.selection.agent_aggressiveness = self.prompts.select(
                message="Coding aggressiveness",
                choices=[("Light", "light"), ("Normal", "normal"), ("Aggressive", "aggressive")],
                default=plan.selection.agent_aggressiveness,
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
