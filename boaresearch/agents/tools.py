from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from .base import ResearchAgentError
from ..schema import CandidateMetadata, OPERATION_TYPES, PATCH_CATEGORIES


@dataclass
class CandidateSubmissionRecorder:
    candidate: Optional[CandidateMetadata] = None

    def submit(
        self,
        *,
        hypothesis: str,
        rationale_summary: str,
        patch_category: str,
        operation_type: str,
        estimated_risk: float,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> None:
        category = str(patch_category).strip().lower()
        operation = str(operation_type).strip().lower()
        if category not in PATCH_CATEGORIES:
            known = ", ".join(sorted(PATCH_CATEGORIES))
            raise ResearchAgentError(f"Unsupported patch_category '{category}'. Expected one of: {known}")
        if operation not in OPERATION_TYPES:
            known = ", ".join(sorted(OPERATION_TYPES))
            raise ResearchAgentError(f"Unsupported operation_type '{operation}'. Expected one of: {known}")
        knobs = numeric_knobs or {}
        if not isinstance(knobs, dict):
            raise ResearchAgentError("numeric_knobs must be a mapping")
        normalized_knobs: dict[str, float] = {}
        for key, value in knobs.items():
            normalized_knobs[str(key)] = float(value)
        self.candidate = CandidateMetadata(
            hypothesis=str(hypothesis).strip(),
            rationale_summary=str(rationale_summary).strip(),
            patch_category=category,
            operation_type=operation,
            estimated_risk=float(estimated_risk),
            target_symbols=[str(item).strip() for item in list(target_symbols or []) if str(item).strip()],
            numeric_knobs=normalized_knobs,
            notes=None if notes is None else str(notes).strip() or None,
        )


class AgentToolHarness:
    def __init__(
        self,
        *,
        run_preflight: Callable[[], None],
        read_recent_trials: Callable[[int], list[dict[str, str]]],
        submission: CandidateSubmissionRecorder,
    ) -> None:
        self._run_preflight = run_preflight
        self._read_recent_trials = read_recent_trials
        self._submission = submission

    def run_preflight(self) -> str:
        try:
            self._run_preflight()
        except Exception as exc:
            return f"Preflight failed: {type(exc).__name__}: {exc}"
        return "Preflight passed."

    def read_recent_trials(self, limit: int = 10) -> str:
        rows = self._read_recent_trials(max(1, int(limit)))
        if not rows:
            return "<no prior trials recorded>"
        return json.dumps(rows, indent=2, sort_keys=True)

    def submit_candidate(
        self,
        hypothesis: str,
        rationale_summary: str,
        patch_category: str,
        operation_type: str,
        estimated_risk: float,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> str:
        self._submission.submit(
            hypothesis=hypothesis,
            rationale_summary=rationale_summary,
            patch_category=patch_category,
            operation_type=operation_type,
            estimated_risk=estimated_risk,
            target_symbols=target_symbols,
            numeric_knobs=numeric_knobs,
            notes=notes,
        )
        return "Candidate metadata recorded."


def _wrap_tool(func, *, name: str, description: str):
    try:
        from langchain_core.tools import tool  # type: ignore
    except Exception as exc:
        raise ResearchAgentError("langchain_core.tools is required for DeepAgents tool wiring") from exc
    return tool(name_or_callable=name, description=description)(func)


def build_agent_tools(harness: AgentToolHarness) -> list[object]:
    return [
        _wrap_tool(harness.run_preflight, name="run_preflight", description="Run the configured preflight commands."),
        _wrap_tool(harness.read_recent_trials, name="read_recent_trials", description="Read recent BOA trial summaries."),
        _wrap_tool(harness.submit_candidate, name="submit_candidate", description="Submit the candidate metadata after edits and preflight are complete."),
    ]
