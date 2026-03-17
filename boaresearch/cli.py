from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .controller import BoaController
from .init_app import InitWizard
from .loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BOA Researcher")
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
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    if args.command == "init":
        InitWizard(initial_path=Path(args.repo)).run()
        return
    if args.command == "run":
        config = load_config(repo=args.repo, config_path=args.config)
        BoaController(config).run()
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
