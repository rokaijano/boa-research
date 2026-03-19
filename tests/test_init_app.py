from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from boaresearch.agent_presets import ModelDiscoveryResult
from boaresearch.init import InitWizard, PromptAdapter
from boaresearch.init.models import (
    DetectedRepo,
    ExistingSetupReport,
    InitSetupSelection,
    PreflightCheck,
    RepoAnalysisProposal,
    ReviewedInitPlan,
    ValidationReport,
    WriteResult,
)
from boaresearch.init import InitServices


class FakePrompts:
    def __init__(self, answers: list[object]) -> None:
        self.answers = list(answers)

    def _next(self):
        if not self.answers:
            raise AssertionError("No prompt answers left")
        return self.answers.pop(0)

    def select(self, **kwargs) -> str:
        return str(self._next())

    def text(self, **kwargs) -> str:
        return str(self._next())

    def confirm(self, **kwargs) -> bool:
        return bool(self._next())


class PromptSpy(FakePrompts):
    def __init__(self, answers: list[object]) -> None:
        super().__init__(answers)
        self.select_calls: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []
        self.confirm_calls: list[dict[str, object]] = []

    def select(self, **kwargs) -> str:
        self.select_calls.append(kwargs)
        return super().select(**kwargs)

    def text(self, **kwargs) -> str:
        self.text_calls.append(kwargs)
        return super().text(**kwargs)

    def confirm(self, **kwargs) -> bool:
        self.confirm_calls.append(kwargs)
        return super().confirm(**kwargs)


class InitAppTests(unittest.TestCase):
    def _services(self, repo_root: Path) -> InitServices:
        detected = DetectedRepo(
            requested_path=repo_root,
            repo_root=repo_root,
            config_path=repo_root / "boa.config",
            boa_md_path=repo_root / "boa.md",
            runtime_root=repo_root / ".boa",
            existing_setup=ExistingSetupReport(status="absent", config_path=repo_root / "boa.config", boa_md_path=repo_root / "boa.md"),
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_command="codex")
        analysis = RepoAnalysisProposal(
            train_command="python train.py",
            eval_command="python eval.py",
            primary_metric_name="accuracy",
            metric_direction="maximize",
            metric_source="regex",
            metric_path=None,
            metric_json_key=None,
            metric_pattern="accuracy=([0-9.]+)",
            editable_files=["src", "tests"],
            protected_files=[".boa/protected"],
            optimization_surfaces=["src"],
            caveats=["Confirm metric extraction"],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = ReviewedInitPlan(
            repo_root=repo_root,
            selection=selection,
            analysis=analysis,
            editable_files=["src", "tests"],
            protected_files=[".boa/protected", ".git"],
            train_command=analysis.train_command,
            eval_command=analysis.eval_command,
            primary_metric_name=analysis.primary_metric_name,
            metric_direction=analysis.metric_direction,
            metric_source=analysis.metric_source,
            metric_path=analysis.metric_path,
            metric_json_key=analysis.metric_json_key,
            metric_pattern=analysis.metric_pattern,
            boa_md=analysis.suggested_boa_md,
        )
        return InitServices(
            detect_repo=lambda path: detected,
            default_selection_for_repo=lambda detected_repo: selection,
            run_preflight=lambda detected_repo, setup: [PreflightCheck("git", True, "git found")],
            analyze_repo=lambda detected_repo, setup: analysis,
            default_repo_analysis=lambda root: analysis,
            merge_reviewed_plan=lambda setup, proposal: plan,
            write_contract_files=lambda reviewed: WriteResult(created_paths=[repo_root / "boa.config", repo_root / "boa.md"]),
            validate_written_setup=lambda reviewed: ValidationReport(passed=True, details=["Config parses successfully."]),
        )

    def test_inquirer_flow_reaches_completion(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = FakePrompts(
            [
                "codex",
                False,
                "default",
                "default",
                "light",
                "local",
                "",
                False,
                True,
            ]
        )
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        draft = wizard.run()
        self.assertIsNotNone(draft.write_result)
        self.assertIsNotNone(draft.validation)
        self.assertEqual(draft.selection.agent_aggressiveness, "light")
        self.assertTrue(any("No BOA files detected" in line for line in output))

    def test_review_mode_stops_before_writing(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        services = self._services(repo_root)
        detected = services.detect_repo(repo_root)
        detected.existing_setup = ExistingSetupReport(
            status="valid",
            config_path=repo_root / "boa.config",
            boa_md_path=repo_root / "boa.md",
            summary="Current setup found",
        )
        prompts = FakePrompts(["review"])
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=InitServices(
                detect_repo=lambda path: detected,
                default_selection_for_repo=services.default_selection_for_repo,
                run_preflight=services.run_preflight,
                analyze_repo=services.analyze_repo,
                default_repo_analysis=services.default_repo_analysis,
                merge_reviewed_plan=services.merge_reviewed_plan,
                write_contract_files=services.write_contract_files,
                validate_written_setup=services.validate_written_setup,
            ),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        draft = wizard.run()
        self.assertIsNone(draft.write_result)

    def test_codex_model_choices_are_listed(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "gpt-5-codex"])
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=lambda _: None,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_command="codex")

        with patch("boaresearch.init.app.shutil.which", return_value="codex"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[("gpt-5.3-codex", "gpt-5.3-codex"), ("gpt-5.1-codex-max", "gpt-5.1-codex-max")],
                    source="codex prompt --models",
                ),
            ):
                wizard._configure_known_cli_agent(selection, preset="codex", default_command="codex")

        model_choices = prompts.select_calls[-1]["choices"]
        self.assertIn(("gpt-5.3-codex", "gpt-5.3-codex"), model_choices)
        self.assertIn(("gpt-5.1-codex-max", "gpt-5.1-codex-max"), model_choices)
        self.assertEqual(selection.agent_model, "gpt-5-codex")

    def test_codex_docs_fallback_choices_are_listed(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "gpt-5.4-mini"])
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_command="codex")

        with patch("boaresearch.init.app.shutil.which", return_value="codex"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[("gpt-5.4", "gpt-5.4"), ("gpt-5.4-mini", "gpt-5.4-mini"), ("gpt-5.3-codex", "gpt-5.3-codex")],
                    source="OpenAI Codex models docs",
                    warning="The installed Codex CLI does not support `codex prompt --models`, so BOA loaded documented Codex models from the public docs page; actual availability may still depend on your account or local configuration.",
                ),
            ):
                with patch(
                    "boaresearch.init.app.check_model_availability_for_agent",
                    return_value=type("Availability", (), {"status": "unknown", "message": None})(),
                ):
                    wizard._configure_known_cli_agent(selection, preset="codex", default_command="codex")

        model_choices = prompts.select_calls[-1]["choices"]
        self.assertIn(("gpt-5.4", "gpt-5.4"), model_choices)
        self.assertIn(("gpt-5.4-mini", "gpt-5.4-mini"), model_choices)
        self.assertIn(("gpt-5.3-codex", "gpt-5.3-codex"), model_choices)
        self.assertTrue(any("public docs page" in line for line in output))
        self.assertEqual(selection.agent_model, "gpt-5.4-mini")

    def test_claude_code_model_choices_are_listed(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "claude-sonnet-4-5"])
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=lambda _: None,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="claude_code", agent_command="claude")

        with patch("boaresearch.init.app.shutil.which", return_value="claude"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[("claude-haiku-4.5", "claude-haiku-4.5"), ("claude-sonnet-4.5", "claude-sonnet-4.5")],
                    source="Anthropic /v1/models",
                ),
            ):
                wizard._configure_known_cli_agent(selection, preset="claude_code", default_command="claude")

        model_choices = prompts.select_calls[-1]["choices"]
        self.assertIn(("claude-haiku-4.5", "claude-haiku-4.5"), model_choices)
        self.assertIn(("claude-sonnet-4.5", "claude-sonnet-4.5"), model_choices)
        self.assertEqual(selection.agent_model, "claude-sonnet-4-5")

    def test_copilot_model_selection_falls_back_to_manual_entry(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "custom", "gpt-5.4-mini"])
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=lambda _: None,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="copilot", agent_command="copilot")

        with patch("boaresearch.init.app.shutil.which", return_value="copilot"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[],
                    warning="Copilot CLI does not expose a non-interactive model list.",
                    supports_listing=False,
                ),
            ):
                wizard._configure_known_cli_agent(selection, preset="copilot", default_command="copilot")

        model_choices = prompts.select_calls[-1]["choices"]
        self.assertEqual(model_choices, [("Use default model", "default"), ("Custom model", "custom")])
        self.assertEqual(selection.agent_model, "gpt-5.4-mini")

    def test_copilot_docs_fallback_choices_are_listed(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "gpt-5.4-mini"])
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="copilot", agent_command="copilot")

        with patch("boaresearch.init.app.shutil.which", return_value="copilot"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[("gpt-5.4", "gpt-5.4"), ("gpt-5.4-mini", "gpt-5.4-mini"), ("claude-sonnet-4.6", "claude-sonnet-4.6")],
                    source="GitHub Copilot supported models docs",
                    warning="Loaded documented Copilot models from GitHub Docs; actual availability depends on your plan, client, and organization policy.",
                ),
            ):
                with patch(
                    "boaresearch.init.app.check_model_availability_for_agent",
                    return_value=type("Availability", (), {"status": "unknown", "message": None})(),
                ):
                    wizard._configure_known_cli_agent(selection, preset="copilot", default_command="copilot")

        model_choices = prompts.select_calls[-1]["choices"]
        self.assertIn(("gpt-5.4", "gpt-5.4"), model_choices)
        self.assertIn(("gpt-5.4-mini", "gpt-5.4-mini"), model_choices)
        self.assertIn(("claude-sonnet-4.6", "claude-sonnet-4.6"), model_choices)
        self.assertTrue(any("GitHub Docs" in line for line in output))
        self.assertEqual(selection.agent_model, "gpt-5.4-mini")

    def test_deepagents_ollama_model_choices_include_available_models(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy(["ollama", "default", "llama3.3:70b"])
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=lambda _: None,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="deepagents", agent_runtime="deepagents")

        with patch(
            "boaresearch.init.app.discover_available_models_for_agent",
            return_value=ModelDiscoveryResult(
                models=[("llama3.3:70b", "llama3.3:70b"), ("qwen2.5-coder:14b", "qwen2.5-coder:14b")],
                source="http://127.0.0.1:11434/api/tags",
            ),
        ):
            wizard._configure_deepagents(selection)

        model_choices = prompts.select_calls[2]["choices"]
        self.assertIn(("llama3.3:70b", "llama3.3:70b"), model_choices)
        self.assertIn(("qwen2.5-coder:14b", "qwen2.5-coder:14b"), model_choices)
        self.assertEqual(selection.agent_model, "llama3.3:70b")

    def test_unavailable_model_is_marked_and_reprompted(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "gpt-5.3-codex", "gpt-5.1-codex-max"])
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_command="codex")

        with patch("boaresearch.init.app.shutil.which", return_value="codex"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[("gpt-5.3-codex", "gpt-5.3-codex"), ("gpt-5.1-codex-max", "gpt-5.1-codex-max")],
                    source="codex prompt --models",
                ),
            ):
                with patch(
                    "boaresearch.init.app.check_model_availability_for_agent",
                    side_effect=[
                        type("Availability", (), {"status": "unavailable", "message": "`gpt-5.3-codex` is not available via codex prompt --models. Choose another model."})(),
                        type("Availability", (), {"status": "available", "message": None})(),
                    ],
                ):
                    wizard._configure_known_cli_agent(selection, preset="codex", default_command="codex")

        self.assertEqual(selection.agent_model, "gpt-5.1-codex-max")
        second_model_prompt = prompts.select_calls[-1]["choices"]
        self.assertIn(("gpt-5.3-codex (not available)", "gpt-5.3-codex"), second_model_prompt)
        self.assertTrue(any("not available" in line for line in output))

    def test_model_availability_status_is_shown_for_non_terminal_outputs(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = PromptSpy([False, "default", "gpt-5.4-mini"])
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="copilot", agent_command="copilot")

        with patch("boaresearch.init.app.shutil.which", return_value="copilot"):
            with patch(
                "boaresearch.init.app.discover_available_models_for_agent",
                return_value=ModelDiscoveryResult(
                    models=[("gpt-5.4-mini", "gpt-5.4-mini")],
                    source="GitHub Copilot supported models docs",
                ),
            ):
                with patch(
                    "boaresearch.init.app.check_model_availability_for_agent",
                    return_value=type("Availability", (), {"status": "available", "message": None})(),
                ):
                    wizard._configure_known_cli_agent(selection, preset="copilot", default_command="copilot")

        self.assertEqual(selection.agent_model, "gpt-5.4-mini")
        self.assertTrue(any(line == "Checking model availability" for line in output))
        self.assertTrue(any(line == "? Model gpt-5.4-mini ✓" for line in output))

    def test_repo_analysis_status_is_shown_for_non_terminal_outputs(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=lambda **_: "defaults", text=lambda **_: "", confirm=lambda **_: False),
            output=output.append,
        )
        wizard.draft.detected_repo = wizard.services.detect_repo(repo_root)
        wizard.draft.selection = wizard.services.default_selection_for_repo(wizard.draft.detected_repo)

        completed = wizard._run_analysis()

        self.assertTrue(completed)
        self.assertTrue(any(line == "Analyzing repo" for line in output))
        self.assertTrue(any("Repo analysis completed." in line for line in output))


if __name__ == "__main__":
    unittest.main()
