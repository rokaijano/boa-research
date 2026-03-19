from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .git_state import GitError
from .init import InitWizard
from .loader import load_config
from .runtime import BoaController
from .runtime.monitor import build_run_observer
from .runtime.controller import ControllerStateError, RunSummary
from .runtime.tools import TOOL_COMMANDS, run_tools_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BOA Research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a repository for BOA")
    init_parser.add_argument("repo", nargs="?", default=".", help="Path inside the target Git repository")

    run_parser = subparsers.add_parser("run", help="Run BOA")
    run_parser.add_argument("repo", nargs="?", default=".", type=Path, help="Path inside the target Git repository")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to boa.config. Defaults to <repo>/boa.config",
    )

    tools_parser = subparsers.add_parser("tools", help="Invoke BOA search tools")
    tools_parser.add_argument("tool_command", choices=sorted(TOOL_COMMANDS.keys()))
    tools_parser.add_argument("repo", nargs="?", default=None, type=Path, help="Optional path inside the target Git repository")
    tools_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to boa.config. Defaults to <repo>/boa.config or BOA_CONFIG_PATH",
    )
    tools_parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="Optional path to a BOA tool context JSON file.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _format_run_summary(summary: RunSummary) -> str:
    if summary.stop_requested:
        if summary.last_trial_id:
            return f"BOA run stopped after {summary.trials_attempted} trial(s). Last trial: {summary.last_trial_id} ({summary.last_acceptance_status or 'unknown'})."
        return "BOA run stopped before starting a new trial."
    if summary.last_trial_id is None:
        return "BOA run completed without creating a trial."
    parts = [
        f"BOA run completed. Last trial: {summary.last_trial_id}",
        f"status={summary.last_acceptance_status or 'unknown'}",
    ]
    if summary.last_canonical_stage:
        parts.append(f"stage={summary.last_canonical_stage}")
    if summary.last_canonical_score is not None:
        parts.append(f"score={summary.last_canonical_score:.6f}")
    parts.append(f"trials_attempted={summary.trials_attempted}")
    if summary.last_detail:
        detail = " ".join(summary.last_detail.split())
        if len(detail) > 220:
            detail = detail[:217].rstrip() + "..."
        parts.append(f"detail={detail}")
    return " | ".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    if args.command == "init":
        try:
            InitWizard(initial_path=Path(args.repo)).run()
        except (KeyboardInterrupt, EOFError):
            print("\nBOA init cancelled. Goodbye.", file=sys.stderr)
        return
    if args.command == "run":
        try:
            config = load_config(repo=args.repo, config_path=args.config)
            with build_run_observer(stream=sys.stderr) as observer:
                summary = BoaController(config, observer=observer).run()
            print(_format_run_summary(summary))
        except (ControllerStateError, GitError) as exc:
            print(f"BOA run failed: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        return
    if args.command == "tools":
        run_tools_command(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
