from __future__ import annotations

import shutil
import subprocess
import unittest
from typing import cast
from unittest.mock import patch

from boaresearch.agent_presets import (
    check_model_availability_for_agent,
    discover_available_models_for_agent,
    discover_codex_models,
    probe_copilot_model_availability,
    probe_codex_model_availability,
)


class ClaudeFallbackDiscoveryTests(unittest.TestCase):
    def test_claude_discovery_falls_back_to_docs_page(self) -> None:
        with patch("boaresearch.agent_presets.discover_anthropic_models", return_value=[]):
            with patch(
                "boaresearch.agent_presets.discover_claude_doc_models",
                return_value=["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
            ):
                discovery = discover_available_models_for_agent(preset="claude_code")

        self.assertEqual(
            discovery.models,
            [
                ("claude-opus-4-6", "claude-opus-4-6"),
                ("claude-sonnet-4-6", "claude-sonnet-4-6"),
                ("claude-haiku-4-5", "claude-haiku-4-5"),
            ],
        )
        self.assertEqual(discovery.source, "Anthropic docs models overview")
        self.assertIsNotNone(discovery.warning)
        warning = cast(str, discovery.warning)
        self.assertIn("public docs page", warning)


class CodexFallbackDiscoveryTests(unittest.TestCase):
    def test_codex_discovery_falls_back_to_docs_page(self) -> None:
        with patch(
            "boaresearch.agent_presets.probe_codex_model_discovery",
            return_value=type(
                "Discovery",
                (),
                {
                    "models": [],
                    "source": "codex prompt --models",
                    "warning": "The installed Codex CLI does not support `codex prompt --models`. You can still enter a model manually.",
                    "supports_listing": True,
                },
            )(),
        ):
            with patch(
                "boaresearch.agent_presets.discover_codex_doc_models",
                return_value=["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
            ):
                discovery = discover_available_models_for_agent(preset="codex", command="codex")

        self.assertEqual(
            discovery.models,
            [
                ("gpt-5.4", "gpt-5.4"),
                ("gpt-5.4-mini", "gpt-5.4-mini"),
                ("gpt-5.3-codex", "gpt-5.3-codex"),
            ],
        )
        self.assertEqual(discovery.source, "OpenAI Codex models docs")
        self.assertIsNotNone(discovery.warning)
        warning = cast(str, discovery.warning)
        self.assertIn("public docs page", warning)


class CopilotFallbackDiscoveryTests(unittest.TestCase):
    def test_copilot_discovery_falls_back_to_docs_page(self) -> None:
        with patch(
            "boaresearch.agent_presets.discover_copilot_doc_models",
            return_value=["gpt-5.4", "gpt-5.4-mini", "claude-sonnet-4.6", "gemini-3-pro"],
        ):
            discovery = discover_available_models_for_agent(preset="copilot", command="copilot")

        self.assertEqual(
            discovery.models,
            [
                ("gpt-5.4", "gpt-5.4"),
                ("gpt-5.4-mini", "gpt-5.4-mini"),
                ("claude-sonnet-4.6", "claude-sonnet-4.6"),
                ("gemini-3-pro", "gemini-3-pro"),
            ],
        )
        self.assertEqual(discovery.source, "GitHub Copilot supported models docs")
        self.assertIsNotNone(discovery.warning)
        warning = cast(str, discovery.warning)
        self.assertIn("GitHub Docs", warning)


class ModelAvailabilityTests(unittest.TestCase):
    def test_copilot_docs_parser_normalizes_display_names(self) -> None:
        page_text = """
        Supported AI models in Copilot
        GPT-5.4
        GPT-5.4 mini
        Claude Sonnet 4.6
        Gemini 3 Pro
        Model retirement history
        GPT-5
        """
        with patch("boaresearch.agent_presets._read_github_docs_article_body", return_value=page_text):
            from boaresearch.agent_presets import discover_copilot_doc_models

            models = discover_copilot_doc_models()

        self.assertEqual(models, ["gpt-5.4", "gpt-5.4-mini", "claude-sonnet-4.6", "gemini-3-pro"])

    def test_copilot_docs_parser_falls_back_to_html_when_article_api_fails(self) -> None:
        page_text = """
        <html><body>
        Supported AI models in Copilot
        <tr><th scope="row">GPT-5.4</th><td>OpenAI</td></tr>
        <tr><th scope="row">GPT-5.4 mini</th><td>OpenAI</td></tr>
        <tr><th scope="row">Claude Opus 4.6 (fast mode) (preview)</th><td>Anthropic</td></tr>
        Model retirement history
        <tr><th scope="row">GPT-5</th><td>OpenAI</td></tr>
        </body></html>
        """
        with patch("boaresearch.agent_presets._read_github_docs_article_body", side_effect=OSError("boom")):
            with patch("boaresearch.agent_presets._read_text_response", return_value=page_text):
                from boaresearch.agent_presets import discover_copilot_doc_models

                models = discover_copilot_doc_models()

        self.assertEqual(models, ["gpt-5.4", "gpt-5.4-mini", "claude-opus-4.6"])

    def test_copilot_probe_reports_unavailable_model(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["copilot"],
            returncode=1,
            stdout='Error: Model "missing-model" from --model flag is not available.',
            stderr="",
        )
        with patch("boaresearch.agent_presets._run_model_discovery_command", return_value=completed):
            result = probe_copilot_model_availability(command="copilot", model="missing-model")

        self.assertEqual(result.status, "unavailable")
        self.assertIsNotNone(result.message)
        message = cast(str, result.message)
        self.assertIn("missing-model", message)

    def test_copilot_probe_reports_available_model(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["copilot"],
            returncode=0,
            stdout='{"type":"assistant.message","data":{"content":"OK"}}',
            stderr="",
        )
        with patch("boaresearch.agent_presets._run_model_discovery_command", return_value=completed):
            result = probe_copilot_model_availability(command="copilot", model="gpt-5.1")

        self.assertEqual(result.status, "available")
        self.assertEqual(result.source, "copilot prompt probe")

    def test_copilot_availability_prefers_probe(self) -> None:
        with patch(
            "boaresearch.agent_presets.probe_copilot_model_availability",
            return_value=type("Availability", (), {"status": "available", "message": None, "source": "copilot prompt probe"})(),
        ):
            result = check_model_availability_for_agent(
                preset="copilot",
                model="gpt-5.1",
                command="copilot",
            )

        self.assertEqual(result.status, "available")
        self.assertEqual(result.source, "copilot prompt probe")

    def test_copilot_availability_uses_longer_probe_timeout_than_listing(self) -> None:
        with patch(
            "boaresearch.agent_presets.probe_copilot_model_availability",
            return_value=type("Availability", (), {"status": "available", "message": None, "source": "copilot prompt probe"})(),
        ) as probe_mock:
            result = check_model_availability_for_agent(
                preset="copilot",
                model="gpt-5-mini",
                command="copilot",
                timeout_seconds=5,
            )

        self.assertEqual(result.status, "available")
        probe_mock.assert_called_once_with(command="copilot", model="gpt-5-mini", timeout_seconds=30)

    def test_copilot_availability_retries_unknown_probe_once(self) -> None:
        with patch(
            "boaresearch.agent_presets.probe_copilot_model_availability",
            side_effect=[
                type("Availability", (), {"status": "unknown", "message": None, "source": "copilot prompt probe"})(),
                type("Availability", (), {"status": "available", "message": None, "source": "copilot prompt probe"})(),
            ],
        ) as probe_mock:
            result = check_model_availability_for_agent(
                preset="copilot",
                model="gpt-5-mini",
                command="copilot",
                timeout_seconds=5,
            )

        self.assertEqual(result.status, "available")
        self.assertEqual(probe_mock.call_args_list[0].kwargs, {"command": "copilot", "model": "gpt-5-mini", "timeout_seconds": 30})
        self.assertEqual(probe_mock.call_args_list[1].kwargs, {"command": "copilot", "model": "gpt-5-mini", "timeout_seconds": 45})

    def test_codex_exec_probe_reports_unavailable_model(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codex"],
            returncode=1,
            stdout='{"type":"error","message":"{\"detail\":\"The \'missing-model\' model is not supported when using Codex with a ChatGPT account.\"}"}',
            stderr="",
        )
        with patch("boaresearch.agent_presets._run_model_discovery_command", return_value=completed):
            result = probe_codex_model_availability(command="codex", model="missing-model")

        self.assertEqual(result.status, "unavailable")
        self.assertIsNotNone(result.message)
        message = cast(str, result.message)
        self.assertIn("missing-model", message)

    def test_codex_exec_probe_reports_available_model(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["codex"],
            returncode=0,
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}',
            stderr="",
        )
        with patch("boaresearch.agent_presets._run_model_discovery_command", return_value=completed):
            result = probe_codex_model_availability(command="codex", model="gpt-5.4")

        self.assertEqual(result.status, "available")
        self.assertEqual(result.source, "codex exec probe")

    def test_codex_exec_probe_uses_inline_prompt_and_closes_stdin(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["stdin"] = kwargs.get("stdin")
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        with patch("boaresearch.agent_presets._command_exists", return_value=True):
            with patch("boaresearch.agent_presets.subprocess.run", side_effect=fake_run):
                result = probe_codex_model_availability(command="codex", model="gpt-5.4")

        self.assertEqual(result.status, "available")
        argv = captured["argv"]
        self.assertIsInstance(argv, list)
        self.assertNotIn("-", argv)
        self.assertIn("Reply with OK only.", argv)
        self.assertEqual(captured["stdin"], subprocess.DEVNULL)

    def test_codex_uses_docs_fallback_for_availability(self) -> None:
        with patch(
            "boaresearch.agent_presets.probe_codex_model_availability",
            return_value=type("Availability", (), {"status": "unknown", "message": None, "source": "codex exec probe"})(),
        ):
            with patch("boaresearch.agent_presets.discover_codex_models", return_value=[]):
                with patch("boaresearch.agent_presets.discover_openai_models", return_value=[]):
                    with patch(
                        "boaresearch.agent_presets.discover_codex_doc_models",
                        return_value=["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"],
                    ):
                        result = check_model_availability_for_agent(
                            preset="codex",
                            model="gpt-5.4-mini",
                            command="codex",
                        )

        self.assertEqual(result.status, "available")
        self.assertEqual(result.source, "OpenAI Codex models docs")

    def test_codex_availability_prefers_exec_probe(self) -> None:
        with patch(
            "boaresearch.agent_presets.probe_codex_model_availability",
            return_value=type("Availability", (), {"status": "available", "message": None, "source": "codex exec probe"})(),
        ):
            with patch("boaresearch.agent_presets.discover_codex_models") as discover_codex_models_mock:
                result = check_model_availability_for_agent(
                    preset="codex",
                    model="gpt-5.4",
                    command="codex",
                )

        self.assertEqual(result.status, "available")
        self.assertEqual(result.source, "codex exec probe")
        discover_codex_models_mock.assert_not_called()

    def test_deepagents_openai_reports_unavailable_model(self) -> None:
        with patch(
            "boaresearch.agent_presets.discover_openai_models",
            return_value=["gpt-5.4", "gpt-5.4-mini"],
        ):
            result = check_model_availability_for_agent(
                preset="deepagents",
                backend="openai",
                model="missing-model",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
            )

        self.assertEqual(result.status, "unavailable")
        self.assertIsNotNone(result.message)
        message = cast(str, result.message)
        self.assertIn("missing-model", message)


@unittest.skipUnless(shutil.which("codex"), "codex is not installed")
class AgentPresetDiscoveryTests(unittest.TestCase):
    def test_codex_model_discovery_matches_live_cli_behavior(self) -> None:
        codex_executable = shutil.which("codex")
        assert codex_executable is not None

        proc = subprocess.run(
            [codex_executable, "prompt", "--models"],
            capture_output=True,
            text=True,
            check=False,
        )
        discovered = discover_codex_models(command="codex")
        discovery = discover_available_models_for_agent(preset="codex", command="codex")

        if proc.returncode == 0:
            self.assertTrue(discovered)
            self.assertEqual(discovered, [value for _, value in discovery.models])
            for model_name in discovered:
                self.assertIn(model_name, proc.stdout)
            self.assertIsNone(discovery.warning)
            return

        self.assertEqual(discovered, [])
        self.assertIn("--models", proc.stderr)
        self.assertTrue(discovery.models or discovery.warning)

    def test_codex_discovery_warning_is_specific_when_cli_lacks_models_command(self) -> None:
        discovery = discover_available_models_for_agent(preset="codex", command="codex")
        if discovery.source == "codex prompt --models" and discovery.models:
            self.assertIsNone(discovery.warning)
            return
        self.assertTrue(discovery.models or discovery.warning)
        if discovery.warning:
            self.assertIn("Codex", discovery.warning)


if __name__ == "__main__":
    unittest.main()
