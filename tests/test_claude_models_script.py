from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "get_claude_models.py"
SPEC = importlib.util.spec_from_file_location("get_claude_models", SCRIPT_PATH)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ClaudeModelsScriptTests(unittest.TestCase):
    def test_extract_claude_models_from_table_rows(self) -> None:
        page_text = """
        Claude API ID | claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5-20251001 |
        Claude API alias | claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5 |
        """
        models = MODULE.extract_claude_models(page_text)
        self.assertEqual(
            models,
            ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-haiku-4-5"],
        )

    def test_extract_claude_models_ignores_bedrock_prefixes(self) -> None:
        page_text = "anthropic.claude-opus-4-6-v1 claude-opus-4-6 claude-haiku-4-5@20251001"
        models = MODULE.extract_claude_models(page_text)
        self.assertEqual(models, ["claude-opus-4-6", "claude-haiku-4-5"])


if __name__ == "__main__":
    unittest.main()