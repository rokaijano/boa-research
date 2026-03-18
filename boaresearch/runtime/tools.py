from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from ..loader import load_config
from ..search import SearchOracleService, SearchToolContext, SearchToolbox, SearchTraceRecorder, read_search_tool_context
from .store import ExperimentStore


TOOL_COMMANDS = {
    "recent-trials": "recent_trials",
    "list-lineage-options": "list_lineage_options",
    "suggest-parents": "suggest_parents",
    "score-candidate-descriptor": "score_candidate_descriptor",
    "rank-patch-families": "rank_patch_families",
    "propose-numeric-knob-regions": "propose_numeric_knob_regions",
}


def _load_request() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Tool request JSON must be an object")
    return dict(payload)


def _manual_context(repo: Path | None, config_path: Path | None) -> SearchToolContext:
    repo_value = repo or Path(os.environ.get("BOA_REPO_ROOT") or ".")
    cfg = load_config(repo=repo_value, config_path=config_path or os.environ.get("BOA_CONFIG_PATH"))
    trace_path = os.environ.get("BOA_SEARCH_TRACE_PATH")
    return SearchToolContext(
        repo_root=cfg.repo_root,
        config_path=cfg.config_path,
        run_tag=cfg.run.tag,
        trial_id=str(os.environ.get("BOA_TRIAL_ID") or "manual"),
        accepted_branch=str(cfg.run.accepted_branch),
        phase=str(os.environ.get("BOA_AGENT_PHASE") or "manual"),
        trace_path=None if not trace_path else Path(trace_path),
    )


def resolve_tool_context(args: Namespace) -> SearchToolContext:
    if getattr(args, "context", None):
        return read_search_tool_context(Path(args.context))
    env_context = os.environ.get("BOA_TOOL_CONTEXT_PATH")
    if env_context:
        path = Path(env_context)
        if path.exists():
            return read_search_tool_context(path)
    return _manual_context(getattr(args, "repo", None), getattr(args, "config", None))


def run_tools_command(args: Namespace) -> int:
    tool_context = resolve_tool_context(args)
    config = load_config(repo=tool_context.repo_root, config_path=tool_context.config_path)
    store = ExperimentStore((config.repo_root / ".boa" / "store" / "experiments.sqlite").resolve())
    store.ensure_schema()
    recent_trials = store.recent_trials(limit=config.search.max_history, run_tag=tool_context.run_tag)
    toolbox = SearchToolbox(
        oracle=SearchOracleService(config=config, memory=recent_trials, accepted_branch=tool_context.accepted_branch),
        recorder=SearchTraceRecorder(trace_path=tool_context.trace_path, phase=tool_context.phase),
    )
    request = _load_request()
    handler_name = TOOL_COMMANDS[str(args.tool_command)]
    handler = getattr(toolbox, handler_name)
    response = handler(request)
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0
