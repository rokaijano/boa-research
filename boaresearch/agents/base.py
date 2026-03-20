from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..schema import (
    AgentExecutionContext,
    AgentPlanningContext,
    AgentReflectionContext,
    CandidateMetadata,
    CandidatePlan,
    OPERATION_TYPES,
    PATCH_CATEGORIES,
    TrialReflection,
)


class ResearchAgentError(RuntimeError):
    pass


class BaseResearchAgent(ABC):
    @abstractmethod
    def plan_trial(self, context: AgentPlanningContext) -> CandidatePlan:
        raise NotImplementedError

    @abstractmethod
    def prepare_candidate(self, context: AgentExecutionContext) -> CandidateMetadata:
        raise NotImplementedError

    @abstractmethod
    def reflect_trial(self, context: AgentReflectionContext) -> TrialReflection:
        raise NotImplementedError


def extract_json_object(text: str, *, preferred_keys: set[str] | None = None) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        candidate = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    matches: list[tuple[int, dict[str, Any]]] = []
    for idx, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(candidate[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            matches.append((idx, obj))
    if matches:
        keys = preferred_keys or {
            "hypothesis",
            "rationale_summary",
            "selected_parent_branch",
            "patch_category",
            "operation_type",
            "estimated_risk",
            "informed_by_call_ids",
        }
        _, best = max(
            matches,
            key=lambda item: (
                sum(1 for key in keys if key in item[1]),
                len(item[1]),
                item[0],
            ),
        )
        return best
    raise ResearchAgentError("Agent output did not contain a valid JSON object")


def _normalize_category_and_operation(data: dict[str, Any]) -> tuple[str, str]:
    patch_category = str(data.get("patch_category", "")).strip().lower()
    if patch_category not in PATCH_CATEGORIES:
        known = ", ".join(sorted(PATCH_CATEGORIES))
        raise ResearchAgentError(f"Unsupported patch_category '{patch_category}'. Expected one of: {known}")
    operation_type = str(data.get("operation_type", "")).strip().lower()
    if operation_type not in OPERATION_TYPES:
        known = ", ".join(sorted(OPERATION_TYPES))
        raise ResearchAgentError(f"Unsupported operation_type '{operation_type}'. Expected one of: {known}")
    return patch_category, operation_type


def _normalize_estimated_risk(data: dict[str, Any]) -> float:
    try:
        return float(data["estimated_risk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchAgentError("estimated_risk must be a number") from exc


def _normalize_numeric_knobs(data: dict[str, Any]) -> dict[str, float]:
    numeric_knobs_raw = data.get("numeric_knobs") or {}
    if not isinstance(numeric_knobs_raw, dict):
        raise ResearchAgentError("numeric_knobs must be a JSON object")
    numeric_knobs: dict[str, float] = {}
    for key, value in numeric_knobs_raw.items():
        try:
            numeric_knobs[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ResearchAgentError(f"numeric_knobs[{key!r}] must be numeric") from exc
    return numeric_knobs


def _normalize_target_symbols(data: dict[str, Any]) -> list[str]:
    symbols = data.get("target_symbols") or []
    if not isinstance(symbols, list):
        raise ResearchAgentError("target_symbols must be a list")
    return [str(symbol).strip() for symbol in symbols if str(symbol).strip()]


def _normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        raise ResearchAgentError(f"{field_name} must be a list")
    if not isinstance(value, list):
        raise ResearchAgentError(f"{field_name} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_informed_by_call_ids(data: dict[str, Any]) -> list[str]:
    raw = data.get("informed_by_call_ids")
    normalized = _normalize_string_list(raw, field_name="informed_by_call_ids")
    if not normalized:
        raise ResearchAgentError("informed_by_call_ids must contain at least one BOA tool call id")
    return normalized


def _normalize_addressed_lesson_ids(data: dict[str, Any]) -> list[str]:
    return _normalize_string_list(data.get("addressed_lesson_ids"), field_name="addressed_lesson_ids")


def parse_candidate_plan_dict(data: dict[str, Any]) -> CandidatePlan:
    missing = [
        key
        for key in (
            "hypothesis",
            "rationale_summary",
            "selected_parent_branch",
            "patch_category",
            "operation_type",
            "estimated_risk",
        )
        if str(data.get(key, "")).strip() == ""
    ]
    if missing:
        raise ResearchAgentError(f"Candidate plan missing required fields: {', '.join(missing)}")
    patch_category, operation_type = _normalize_category_and_operation(data)
    return CandidatePlan(
        hypothesis=str(data["hypothesis"]).strip(),
        rationale_summary=str(data["rationale_summary"]).strip(),
        selected_parent_branch=str(data["selected_parent_branch"]).strip(),
        selected_parent_trial_id=None
        if data.get("selected_parent_trial_id") in {None, ""}
        else str(data.get("selected_parent_trial_id")).strip(),
        patch_category=patch_category,
        operation_type=operation_type,
        estimated_risk=_normalize_estimated_risk(data),
        target_symbols=_normalize_target_symbols(data),
        numeric_knobs=_normalize_numeric_knobs(data),
        notes=None if data.get("notes") is None else str(data.get("notes")).strip() or None,
        informed_by_call_ids=_normalize_informed_by_call_ids(data),
        addressed_lesson_ids=_normalize_addressed_lesson_ids(data),
    )


def parse_candidate_dict(data: dict[str, Any]) -> CandidateMetadata:
    missing = [
        key
        for key in ("hypothesis", "rationale_summary", "patch_category", "operation_type", "estimated_risk")
        if str(data.get(key, "")).strip() == ""
    ]
    if missing:
        raise ResearchAgentError(f"Candidate metadata missing required fields: {', '.join(missing)}")
    patch_category, operation_type = _normalize_category_and_operation(data)
    return CandidateMetadata(
        hypothesis=str(data["hypothesis"]).strip(),
        rationale_summary=str(data["rationale_summary"]).strip(),
        patch_category=patch_category,
        operation_type=operation_type,
        estimated_risk=_normalize_estimated_risk(data),
        target_symbols=_normalize_target_symbols(data),
        numeric_knobs=_normalize_numeric_knobs(data),
        notes=None if data.get("notes") is None else str(data.get("notes")).strip() or None,
        informed_by_call_ids=_normalize_informed_by_call_ids(data),
        addressed_lesson_ids=_normalize_addressed_lesson_ids(data),
    )


def parse_trial_reflection_dict(data: dict[str, Any]) -> TrialReflection:
    source_stage = str(data.get("source_stage", "")).strip()
    behavior_summary = str(data.get("behavior_summary", "")).strip()
    primary_problem = str(data.get("primary_problem", "")).strip()
    outcome = str(data.get("outcome", "")).strip()
    missing = [
        key
        for key, value in (
            ("source_stage", source_stage),
            ("behavior_summary", behavior_summary),
            ("primary_problem", primary_problem),
            ("outcome", outcome),
        )
        if not value
    ]
    if missing:
        raise ResearchAgentError(f"Trial reflection missing required fields: {', '.join(missing)}")
    return TrialReflection(
        source_stage=source_stage,
        source_commands=_normalize_string_list(data.get("source_commands"), field_name="source_commands"),
        behavior_summary=behavior_summary,
        primary_problem=primary_problem,
        under_optimized=_normalize_string_list(data.get("under_optimized"), field_name="under_optimized"),
        suggested_fixes=_normalize_string_list(data.get("suggested_fixes"), field_name="suggested_fixes"),
        evidence=_normalize_string_list(data.get("evidence"), field_name="evidence"),
        outcome=outcome,
    )


@dataclass
class AgentAdapterContext:
    planning: AgentPlanningContext | None = None
    execution: AgentExecutionContext | None = None
