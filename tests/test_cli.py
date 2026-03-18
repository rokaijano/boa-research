from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from boaresearch.cli import build_parser, main


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

    def test_tools_accept_subcommand_and_optional_paths(self) -> None:
        args = build_parser().parse_args(["tools", "recent-trials", "/tmp/repo", "--config", "/tmp/repo/boa.config"])
        self.assertEqual(args.command, "tools")
        self.assertEqual(args.tool_command, "recent-trials")
        self.assertEqual(args.repo, Path("/tmp/repo"))
        self.assertEqual(args.config, Path("/tmp/repo/boa.config"))

    def test_init_interrupt_prints_goodbye_message(self) -> None:
        stderr = io.StringIO()
        with patch("boaresearch.cli.parse_args", return_value=build_parser().parse_args(["init"])):
            with patch("boaresearch.cli.InitWizard") as wizard_cls:
                wizard_cls.return_value.run.side_effect = KeyboardInterrupt()
                with patch("sys.stderr", stderr):
                    main()
        self.assertIn("BOA init cancelled. Goodbye.", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
