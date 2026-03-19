from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "get_codex_models.py"
SPEC = importlib.util.spec_from_file_location("get_codex_models", SCRIPT_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CodexModelsScriptTests(unittest.TestCase):
    def test_extract_codex_models_from_page_snippets(self) -> None:
        page_text = """
        codex -m gpt-5.4
        codex -m gpt-5.4-mini
        codex -m gpt-5.3-codex
        codex -m gpt-5.3-codex
        codex -m gpt-5.1-codex-max
        """
        models = MODULE.extract_codex_models(page_text)
        self.assertEqual(
            models,
            ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.1-codex-max"],
        )

    def test_extract_codex_models_handles_html_entities(self) -> None:
        page_text = "codex -m gpt-5.4&amp;nbsp; codex -m gpt-5.2-codex"
        models = MODULE.extract_codex_models(page_text)
        self.assertEqual(models, ["gpt-5.4", "gpt-5.2-codex"])


if __name__ == "__main__":
    unittest.main()