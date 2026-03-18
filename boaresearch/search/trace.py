from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..schema import SearchToolCall


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_id() -> str:
    return f"boa-call-{uuid.uuid4().hex[:12]}"


@dataclass
class SearchToolContext:
    repo_root: Path
    config_path: Path
    run_tag: str
    trial_id: str
    accepted_branch: str
    phase: str
    trace_path: Optional[Path] = None


class SearchTraceRecorder:
    def __init__(self, *, trace_path: Path | None, phase: str) -> None:
        self.trace_path = trace_path
        self.phase = phase

    def record(self, *, tool_name: str, request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        payload = dict(response)
        payload["call_id"] = call_id()
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            record = SearchToolCall(
                call_id=payload["call_id"],
                tool_name=tool_name,
                phase=self.phase,
                request=request,
                response=response,
                created_at=utc_now(),
            )
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return payload


def write_search_tool_context(path: Path, context: SearchToolContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_root": str(context.repo_root),
        "config_path": str(context.config_path),
        "run_tag": context.run_tag,
        "trial_id": context.trial_id,
        "accepted_branch": context.accepted_branch,
        "phase": context.phase,
        "trace_path": None if context.trace_path is None else str(context.trace_path),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_search_tool_context(path: Path) -> SearchToolContext:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SearchToolContext(
        repo_root=Path(data["repo_root"]),
        config_path=Path(data["config_path"]),
        run_tag=str(data["run_tag"]),
        trial_id=str(data["trial_id"]),
        accepted_branch=str(data["accepted_branch"]),
        phase=str(data["phase"]),
        trace_path=None if not data.get("trace_path") else Path(str(data["trace_path"])),
    )


def load_search_trace(path: Path) -> list[SearchToolCall]:
    if not path.exists():
        return []
    calls: list[SearchToolCall] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        calls.append(
            SearchToolCall(
                call_id=str(payload["call_id"]),
                tool_name=str(payload["tool_name"]),
                phase=str(payload["phase"]),
                request=dict(payload["request"]),
                response=dict(payload["response"]),
                created_at=str(payload["created_at"]),
            )
        )
    return calls
