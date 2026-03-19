from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from .base import ResearchAgentError
from .interaction import BoaInteractionLayer
from ..schema import CandidateMetadata, CandidatePlan
from ..search import SearchToolbox


@dataclass
class CandidatePlanSubmissionRecorder:
    plan: Optional[CandidatePlan] = None
    interaction: BoaInteractionLayer = BoaInteractionLayer()

    def submit(
        self,
        *,
        hypothesis: str,
        rationale_summary: str,
        selected_parent_branch: str,
        selected_parent_trial_id: Optional[str] = None,
        patch_category: str,
        operation_type: str,
        estimated_risk: float,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        notes: Optional[str] = None,
        informed_by_call_ids: Optional[list[str]] = None,
    ) -> None:
        self.plan = self.interaction.parse_plan_payload(
            {
                "hypothesis": hypothesis,
                "rationale_summary": rationale_summary,
                "selected_parent_branch": selected_parent_branch,
                "selected_parent_trial_id": selected_parent_trial_id,
                "patch_category": patch_category,
                "operation_type": operation_type,
                "estimated_risk": estimated_risk,
                "target_symbols": target_symbols or [],
                "numeric_knobs": numeric_knobs or {},
                "notes": notes,
                "informed_by_call_ids": informed_by_call_ids or [],
            }
        )


@dataclass
class CandidateSubmissionRecorder:
    candidate: Optional[CandidateMetadata] = None
    interaction: BoaInteractionLayer = BoaInteractionLayer()

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
        informed_by_call_ids: Optional[list[str]] = None,
    ) -> None:
        self.candidate = self.interaction.parse_candidate_payload(
            {
                "hypothesis": hypothesis,
                "rationale_summary": rationale_summary,
                "patch_category": patch_category,
                "operation_type": operation_type,
                "estimated_risk": estimated_risk,
                "target_symbols": target_symbols or [],
                "numeric_knobs": numeric_knobs or {},
                "notes": notes,
                "informed_by_call_ids": informed_by_call_ids or [],
            }
        )


class AgentToolHarness:
    def __init__(
        self,
        *,
        run_preflight: Callable[[], None],
        search_tools: SearchToolbox,
        plan_submission: CandidatePlanSubmissionRecorder | None = None,
        candidate_submission: CandidateSubmissionRecorder | None = None,
    ) -> None:
        self._run_preflight = run_preflight
        self._search_tools = search_tools
        self._plan_submission = plan_submission
        self._candidate_submission = candidate_submission

    @staticmethod
    def _json_text(payload: dict[str, object]) -> str:
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def _descriptor_request(
        *,
        patch_category: str = "misc",
        operation_type: str = "replace",
        estimated_risk: float = 0.25,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        touched_files: Optional[list[str]] = None,
        parent_branch: Optional[str] = None,
        parent_trial_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "patch_category": patch_category,
            "operation_type": operation_type,
            "estimated_risk": estimated_risk,
            "target_symbols": list(target_symbols or []),
            "numeric_knobs": dict(numeric_knobs or {}),
            "touched_files": list(touched_files or []),
        }
        if parent_branch:
            payload["parent_branch"] = parent_branch
        if parent_trial_id:
            payload["parent_trial_id"] = parent_trial_id
        if limit is not None:
            payload["limit"] = int(limit)
        return payload

    def run_preflight(self) -> str:
        try:
            self._run_preflight()
        except Exception as exc:
            return f"Preflight failed: {type(exc).__name__}: {exc}"
        return "Preflight passed."

    def recent_trials(self, limit: int = 10) -> str:
        return self._json_text(self._search_tools.recent_trials({"limit": max(1, int(limit))}))

    def list_lineage_options(self, limit: int = 10) -> str:
        return self._json_text(self._search_tools.list_lineage_options({"limit": max(1, int(limit))}))

    def suggest_parents(
        self,
        patch_category: str = "misc",
        operation_type: str = "replace",
        estimated_risk: float = 0.25,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        touched_files: Optional[list[str]] = None,
        parent_branch: Optional[str] = None,
        parent_trial_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        return self._json_text(
            self._search_tools.suggest_parents(
                self._descriptor_request(
                    patch_category=patch_category,
                    operation_type=operation_type,
                    estimated_risk=estimated_risk,
                    target_symbols=target_symbols,
                    numeric_knobs=numeric_knobs,
                    touched_files=touched_files,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                    limit=limit,
                )
            )
        )

    def score_candidate_descriptor(
        self,
        patch_category: str,
        operation_type: str,
        estimated_risk: float,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        touched_files: Optional[list[str]] = None,
        parent_branch: Optional[str] = None,
        parent_trial_id: Optional[str] = None,
    ) -> str:
        return self._json_text(
            self._search_tools.score_candidate_descriptor(
                self._descriptor_request(
                    patch_category=patch_category,
                    operation_type=operation_type,
                    estimated_risk=estimated_risk,
                    target_symbols=target_symbols,
                    numeric_knobs=numeric_knobs,
                    touched_files=touched_files,
                    parent_branch=parent_branch,
                    parent_trial_id=parent_trial_id,
                )
            )
        )

    def rank_patch_families(
        self,
        operation_type: str = "replace",
        estimated_risk: float = 0.25,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        touched_files: Optional[list[str]] = None,
        parent_branch: Optional[str] = None,
        parent_trial_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        request = self._descriptor_request(
            operation_type=operation_type,
            estimated_risk=estimated_risk,
            target_symbols=target_symbols,
            numeric_knobs=numeric_knobs,
            touched_files=touched_files,
            parent_branch=parent_branch,
            parent_trial_id=parent_trial_id,
            limit=limit,
        )
        request.pop("patch_category", None)
        return self._json_text(self._search_tools.rank_patch_families(request))

    def propose_numeric_knob_regions(
        self,
        patch_category: str = "misc",
        operation_type: str = "replace",
        estimated_risk: float = 0.25,
        numeric_knobs: Optional[dict[str, float]] = None,
        parent_branch: Optional[str] = None,
        parent_trial_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        request = self._descriptor_request(
            patch_category=patch_category,
            operation_type=operation_type,
            estimated_risk=estimated_risk,
            numeric_knobs=numeric_knobs,
            parent_branch=parent_branch,
            parent_trial_id=parent_trial_id,
            limit=limit,
        )
        request.pop("target_symbols", None)
        request.pop("touched_files", None)
        return self._json_text(self._search_tools.propose_numeric_knob_regions(request))

    def submit_candidate_plan(
        self,
        hypothesis: str,
        rationale_summary: str,
        selected_parent_branch: str,
        patch_category: str,
        operation_type: str,
        estimated_risk: float,
        informed_by_call_ids: list[str],
        selected_parent_trial_id: Optional[str] = None,
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> str:
        if self._plan_submission is None:
            raise ResearchAgentError("submit_candidate_plan is not available in this phase")
        self._plan_submission.submit(
            hypothesis=hypothesis,
            rationale_summary=rationale_summary,
            selected_parent_branch=selected_parent_branch,
            selected_parent_trial_id=selected_parent_trial_id,
            patch_category=patch_category,
            operation_type=operation_type,
            estimated_risk=estimated_risk,
            target_symbols=target_symbols,
            numeric_knobs=numeric_knobs,
            notes=notes,
            informed_by_call_ids=informed_by_call_ids,
        )
        return "Candidate plan recorded."

    def submit_candidate(
        self,
        hypothesis: str,
        rationale_summary: str,
        patch_category: str,
        operation_type: str,
        estimated_risk: float,
        informed_by_call_ids: list[str],
        target_symbols: Optional[list[str]] = None,
        numeric_knobs: Optional[dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> str:
        if self._candidate_submission is None:
            raise ResearchAgentError("submit_candidate is not available in this phase")
        self._candidate_submission.submit(
            hypothesis=hypothesis,
            rationale_summary=rationale_summary,
            patch_category=patch_category,
            operation_type=operation_type,
            estimated_risk=estimated_risk,
            target_symbols=target_symbols,
            numeric_knobs=numeric_knobs,
            notes=notes,
            informed_by_call_ids=informed_by_call_ids,
        )
        return "Candidate metadata recorded."


def _wrap_tool(func, *, name: str, description: str):
    try:
        from langchain_core.tools import tool  # type: ignore
    except Exception as exc:
        raise ResearchAgentError("langchain_core.tools is required for DeepAgents tool wiring") from exc
    return tool(name_or_callable=name, description=description)(func)


def build_agent_tools(harness: AgentToolHarness, *, phase: str) -> list[object]:
    tools = [
        _wrap_tool(harness.recent_trials, name="recent_trials", description="Read recent BOA trial summaries."),
        _wrap_tool(
            harness.list_lineage_options,
            name="list_lineage_options",
            description="List BOA-known parent lineage options for the current run.",
        ),
        _wrap_tool(
            harness.suggest_parents,
            name="suggest_parents",
            description="Rank parent branches using the BOA Bayesian oracle.",
        ),
        _wrap_tool(
            harness.score_candidate_descriptor,
            name="score_candidate_descriptor",
            description="Score a draft patch descriptor with the BOA Bayesian oracle.",
        ),
        _wrap_tool(
            harness.rank_patch_families,
            name="rank_patch_families",
            description="Rank patch families using BOA search memory.",
        ),
        _wrap_tool(
            harness.propose_numeric_knob_regions,
            name="propose_numeric_knob_regions",
            description="Suggest numeric knob regions from BOA trial memory.",
        ),
    ]
    if phase == "planning":
        tools.append(
            _wrap_tool(
                harness.submit_candidate_plan,
                name="submit_candidate_plan",
                description="Submit the candidate plan after selecting a parent branch and BO-informed strategy.",
            )
        )
        return tools
    tools.extend(
        [
            _wrap_tool(harness.run_preflight, name="run_preflight", description="Run the configured preflight commands."),
            _wrap_tool(
                harness.submit_candidate,
                name="submit_candidate",
                description="Submit the final candidate metadata after edits and preflight are complete.",
            ),
        ]
    )
    return tools
