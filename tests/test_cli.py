from __future__ import annotations

import unittest
from pathlib import Path

from boaresearch.cli import build_parser


class CliTests(unittest.TestCase):
    def test_init_defaults_to_current_directory(self) -> None:
        args = build_parser().parse_args(["init"])
        self.assertEqual(args.command, "init")
        self.assertEqual(args.repo, ".")

    def test_init_accepts_positional_repo(self) -> None:
        args = build_parser().parse_args(["init", "/tmp/repo"])
        self.assertEqual(args.command, "init")
        self.assertEqual(args.repo, "/tmp/repo")

    def test_run_defaults_to_current_directory(self) -> None:
        args = build_parser().parse_args(["run"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.repo, Path("."))

    def test_run_accepts_positional_repo_and_config(self) -> None:
        args = build_parser().parse_args(["run", "/tmp/repo", "--config", "/tmp/repo/boa.config"])
        self.assertEqual(args.command, "run")
        self.assertEqual(args.repo, Path("/tmp/repo"))
        self.assertEqual(args.config, Path("/tmp/repo/boa.config"))


if __name__ == "__main__":
    unittest.main()
