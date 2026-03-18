from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class PackagingTests(unittest.TestCase):
    def test_prompt_templates_are_declared_as_package_data(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        package_data = pyproject["tool"]["setuptools"]["package-data"]["boaresearch"]
        manifest = (repo_root / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("prompts/**/*.md", package_data)
        self.assertIn("recursive-include boaresearch/prompts *.md", manifest)
