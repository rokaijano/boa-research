from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..schema import AgentContext, CandidateMetadata, OPERATION_TYPES, PATCH_CATEGORIES


class ResearchAgentError(RuntimeError):
    pass


class BaseResearchAgent(ABC):
    @abstractmethod
    def prepare_candidate(self, context: AgentContext) -> CandidateMetadata:
        raise NotImplementedError


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(candidate[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ResearchAgentError("Agent output did not contain a valid JSON object")


def parse_candidate_dict(data: dict[str, Any]) -> CandidateMetadata:
    missing = [
        key
        for key in ("hypothesis", "rationale_summary", "patch_category", "operation_type", "estimated_risk")
        if str(data.get(key, "")).strip() == ""
    ]
    if missing:
        raise ResearchAgentError(f"Candidate metadata missing required fields: {', '.join(missing)}")
    patch_category = str(data["patch_category"]).strip().lower()
    if patch_category not in PATCH_CATEGORIES:
        known = ", ".join(sorted(PATCH_CATEGORIES))
        raise ResearchAgentError(f"Unsupported patch_category '{patch_category}'. Expected one of: {known}")
    operation_type = str(data["operation_type"]).strip().lower()
    if operation_type not in OPERATION_TYPES:
        known = ", ".join(sorted(OPERATION_TYPES))
        raise ResearchAgentError(f"Unsupported operation_type '{operation_type}'. Expected one of: {known}")
    try:
        estimated_risk = float(data["estimated_risk"])
    except (TypeError, ValueError) as exc:
        raise ResearchAgentError("estimated_risk must be a number") from exc
    numeric_knobs_raw = data.get("numeric_knobs") or {}
    if not isinstance(numeric_knobs_raw, dict):
        raise ResearchAgentError("numeric_knobs must be a JSON object")
    numeric_knobs: dict[str, float] = {}
    for key, value in numeric_knobs_raw.items():
        try:
            numeric_knobs[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ResearchAgentError(f"numeric_knobs[{key!r}] must be numeric") from exc
    symbols = data.get("target_symbols") or []
    if not isinstance(symbols, list):
        raise ResearchAgentError("target_symbols must be a list")
    return CandidateMetadata(
        hypothesis=str(data["hypothesis"]).strip(),
        rationale_summary=str(data["rationale_summary"]).strip(),
        patch_category=patch_category,
        operation_type=operation_type,
        estimated_risk=estimated_risk,
        target_symbols=[str(symbol).strip() for symbol in symbols if str(symbol).strip()],
        numeric_knobs=numeric_knobs,
        notes=None if data.get("notes") is None else str(data.get("notes")).strip() or None,
    )


@dataclass
class AgentAdapterContext:
    context: AgentContext
