from __future__ import annotations

from contextlib import nullcontext
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from boaresearch.cli import build_parser, main
from boaresearch.runtime.controller import ControllerStateError, RunSummary


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

    def test_run_repository_state_error_prints_helpful_message(self) -> None:
        stderr = io.StringIO()
        with patch("boaresearch.cli.parse_args", return_value=build_parser().parse_args(["run"])):
            with patch("boaresearch.cli.load_config", return_value=object()):
                with patch("boaresearch.cli.BoaController") as controller_cls:
                    with patch("boaresearch.cli.build_run_observer", return_value=nullcontext(object())):
                        controller_cls.return_value.run.side_effect = ControllerStateError("Create an initial commit.")
                        with patch("sys.stderr", stderr):
                            with self.assertRaises(SystemExit) as exit_ctx:
                                main()
        self.assertEqual(exit_ctx.exception.code, 2)
        self.assertIn("BOA run failed: Create an initial commit.", stderr.getvalue())

    def test_run_prints_summary(self) -> None:
        stdout = io.StringIO()
        with patch("boaresearch.cli.parse_args", return_value=build_parser().parse_args(["run"])):
            with patch("boaresearch.cli.load_config", return_value=object()):
                with patch("boaresearch.cli.BoaController") as controller_cls:
                    with patch("boaresearch.cli.build_run_observer", return_value=nullcontext(object())):
                        controller_cls.return_value.run.return_value = RunSummary(
                            trials_attempted=1,
                            stop_requested=False,
                            last_trial_id="demo-0001",
                            last_acceptance_status="agent_failed",
                            last_canonical_stage=None,
                            last_canonical_score=None,
                            last_detail="CLI agent 'codex' exited with code 1 during planning.",
                        )
                        with patch("sys.stdout", stdout):
                            main()
        self.assertIn(
            "BOA run completed. Last trial: demo-0001 | status=agent_failed | trials_attempted=1 | detail=CLI agent 'codex' exited with code 1 during planning.",
            stdout.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
