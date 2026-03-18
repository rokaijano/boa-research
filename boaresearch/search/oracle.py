from __future__ import annotations

from typing import Any, Optional

from ..schema import BoaConfig, DescriptorDraft, PATCH_CATEGORIES, PatchDescriptor, TrialSummary
from .scoring import descriptor_trials, gaussian_similarity, gp_posterior


def summary(trial: TrialSummary) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "run_tag": trial.run_tag,
        "branch_name": trial.branch_name,
        "parent_branch": trial.parent_branch,
        "parent_trial_id": trial.parent_trial_id,
        "acceptance_status": trial.acceptance_status,
        "canonical_stage": trial.canonical_stage,
        "canonical_score": trial.canonical_score,
        "patch_category": trial.descriptor.patch_category if trial.descriptor else None,
        "rationale_summary": trial.candidate.rationale_summary if trial.candidate else None,
    }


class SearchOracleService:
    def __init__(self, *, config: BoaConfig, memory: list[TrialSummary], accepted_branch: str) -> None:
        self.config = config
        self.memory = [trial for trial in memory if trial.run_tag == config.run.tag]
        self.accepted_branch = accepted_branch

    def recent_trials(self, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
        rows = self.memory[: max(1, int(limit or self.config.search.max_history))]
        return [summary(trial) for trial in rows]

    def list_lineage_options(self, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
        count = max(1, int(limit or self.config.search.max_history))
        options = [
            {
                "branch_name": self.accepted_branch,
                "trial_id": None,
                "acceptance_status": "accepted_branch",
                "canonical_stage": None,
                "canonical_score": None,
                "patch_category": None,
                "rationale_summary": "Current accepted branch.",
            }
        ]
        for trial in self.memory[:count]:
            options.append(
                {
                    "branch_name": trial.branch_name,
                    "trial_id": trial.trial_id,
                    "acceptance_status": trial.acceptance_status,
                    "canonical_stage": trial.canonical_stage,
                    "canonical_score": trial.canonical_score,
                    "patch_category": trial.descriptor.patch_category if trial.descriptor else None,
                    "rationale_summary": trial.candidate.rationale_summary if trial.candidate else None,
                }
            )
        return options

    def validate_parent_selection(self, *, branch_name: str, trial_id: str | None) -> dict[str, Any] | None:
        if branch_name == self.accepted_branch:
            if trial_id not in {None, ""}:
                return None
            return {
                "branch_name": self.accepted_branch,
                "trial_id": None,
                "source": "accepted_branch",
            }
        for option in self.list_lineage_options():
            if option["branch_name"] != branch_name:
                continue
            if trial_id not in {None, "", option["trial_id"]}:
                continue
            return {
                "branch_name": option["branch_name"],
                "trial_id": option["trial_id"],
                "source": "trial",
            }
        return None

    def observed_descriptors(self) -> list[PatchDescriptor]:
        return [trial.descriptor for trial in descriptor_trials(self.memory) if trial.descriptor is not None]

    def observed_scores(self) -> list[float]:
        return [float(trial.canonical_score or 0.0) for trial in descriptor_trials(self.memory)]

    def draft_to_descriptor(self, draft: DescriptorDraft) -> PatchDescriptor:
        parent_branch = draft.parent_branch or self.accepted_branch
        return PatchDescriptor(
            touched_files=list(draft.touched_files),
            touched_symbols=list(draft.target_symbols),
            patch_category=draft.patch_category,
            operation_type=draft.operation_type,
            numeric_knobs=dict(draft.numeric_knobs),
            rationale_summary="oracle_draft",
            estimated_risk=float(draft.estimated_risk),
            parent_branch=parent_branch,
            parent_trial_id=draft.parent_trial_id,
            budget_used="oracle",
            diff_path="",
        )

    def baseline_score(self, descriptor: PatchDescriptor) -> dict[str, float]:
        mean = 0.0
        stddev = 1.0
        acquisition = mean + float(self.config.search.exploration_weight) * stddev
        acquisition -= float(self.config.search.risk_penalty) * float(descriptor.estimated_risk)
        if descriptor.patch_category != "misc":
            acquisition += float(self.config.search.family_bonus)
        if descriptor.parent_trial_id:
            acquisition += float(self.config.search.lineage_bonus)
        return {
            "acquisition_score": acquisition,
            "posterior_mean": mean,
            "posterior_std": stddev,
        }

    def score_descriptor(self, descriptor: PatchDescriptor) -> dict[str, float]:
        observed = self.observed_descriptors()
        if len(observed) < 2:
            return self.baseline_score(descriptor)
        mean, stddev = gp_posterior(
            observed,
            self.observed_scores(),
            descriptor,
            noise=float(self.config.search.observation_noise),
        )
        acquisition = mean + float(self.config.search.exploration_weight) * stddev
        acquisition -= float(self.config.search.risk_penalty) * float(descriptor.estimated_risk)
        if descriptor.patch_category != "misc":
            acquisition += float(self.config.search.family_bonus)
        if descriptor.parent_trial_id:
            acquisition += float(self.config.search.lineage_bonus)
        return {
            "acquisition_score": acquisition,
            "posterior_mean": mean,
            "posterior_std": stddev,
        }

    def parse_descriptor_request(self, request: dict[str, Any]) -> DescriptorDraft:
        knobs_raw = request.get("numeric_knobs") or {}
        touched_raw = request.get("touched_files") or []
        symbols_raw = request.get("target_symbols") or []
        return DescriptorDraft(
            patch_category=str(request.get("patch_category") or "misc").strip().lower(),
            operation_type=str(request.get("operation_type") or "replace").strip().lower(),
            estimated_risk=float(request.get("estimated_risk", 0.25)),
            target_symbols=[str(item) for item in list(symbols_raw)],
            numeric_knobs={str(key): float(value) for key, value in dict(knobs_raw).items()},
            touched_files=[str(item) for item in list(touched_raw)],
            parent_branch=None if not request.get("parent_branch") else str(request["parent_branch"]),
            parent_trial_id=None if not request.get("parent_trial_id") else str(request["parent_trial_id"]),
        )

    def score_candidate_descriptor(self, request: dict[str, Any]) -> dict[str, Any]:
        draft = self.parse_descriptor_request(request)
        descriptor = self.draft_to_descriptor(draft)
        scored = self.score_descriptor(descriptor)
        return {
            "descriptor": {
                "patch_category": draft.patch_category,
                "operation_type": draft.operation_type,
                "estimated_risk": draft.estimated_risk,
                "target_symbols": draft.target_symbols,
                "numeric_knobs": draft.numeric_knobs,
                "touched_files": draft.touched_files,
                "parent_branch": draft.parent_branch or self.accepted_branch,
                "parent_trial_id": draft.parent_trial_id,
            },
            "observation_count": len(self.observed_descriptors()),
            "family_bonus": float(self.config.search.family_bonus) if draft.patch_category != "misc" else 0.0,
            "lineage_bonus": float(self.config.search.lineage_bonus) if draft.parent_trial_id else 0.0,
            "risk_penalty": float(self.config.search.risk_penalty) * float(draft.estimated_risk),
            **scored,
        }

    def suggest_parents(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        observed = descriptor_trials(self.memory)
        count = max(1, int(request.get("limit") or self.config.search.parent_suggestion_count))
        draft: DescriptorDraft | None = None
        if request:
            draft = self.parse_descriptor_request(request)
        suggestions: list[dict[str, Any]] = []
        if len(observed) < 2:
            return [
                {
                    "branch_name": self.accepted_branch,
                    "trial_id": None,
                    "source": "accepted_branch",
                    "reason": "Warm start from the accepted branch while BO memory is sparse.",
                    "acquisition_score": self.baseline_score(
                        self.draft_to_descriptor(
                            draft or DescriptorDraft(
                                patch_category="misc",
                                operation_type="replace",
                                estimated_risk=0.25,
                            )
                        )
                    )["acquisition_score"],
                }
            ]
        draft_descriptor = None if draft is None else self.draft_to_descriptor(draft)
        for trial in observed:
            assert trial.descriptor is not None
            parent_descriptor = trial.descriptor
            candidate_descriptor = parent_descriptor
            similarity = None
            if draft_descriptor is not None:
                candidate_descriptor = PatchDescriptor(
                    touched_files=list(draft_descriptor.touched_files),
                    touched_symbols=list(draft_descriptor.touched_symbols),
                    patch_category=draft_descriptor.patch_category,
                    operation_type=draft_descriptor.operation_type,
                    numeric_knobs=dict(draft_descriptor.numeric_knobs),
                    rationale_summary=draft_descriptor.rationale_summary,
                    estimated_risk=draft_descriptor.estimated_risk,
                    parent_branch=trial.branch_name,
                    parent_trial_id=trial.trial_id,
                    budget_used=draft_descriptor.budget_used,
                    diff_path=draft_descriptor.diff_path,
                )
                similarity = gaussian_similarity(parent_descriptor, draft_descriptor)
            scored = self.score_descriptor(candidate_descriptor)
            suggestions.append(
                {
                    "branch_name": trial.branch_name,
                    "trial_id": trial.trial_id,
                    "source": "trial",
                    "patch_category": trial.descriptor.patch_category,
                    "canonical_score": trial.canonical_score,
                    "posterior_mean": scored["posterior_mean"],
                    "posterior_std": scored["posterior_std"],
                    "acquisition_score": scored["acquisition_score"] + (0.2 * similarity if similarity is not None else 0.0),
                    "similarity_to_request": similarity,
                    "reason": trial.candidate.rationale_summary if trial.candidate else "",
                }
            )
        suggestions.sort(key=lambda item: float(item["acquisition_score"]), reverse=True)
        accepted_scored: dict[str, float] | None = None
        if draft_descriptor is not None:
            accepted_descriptor = PatchDescriptor(
                touched_files=list(draft_descriptor.touched_files),
                touched_symbols=list(draft_descriptor.touched_symbols),
                patch_category=draft_descriptor.patch_category,
                operation_type=draft_descriptor.operation_type,
                numeric_knobs=dict(draft_descriptor.numeric_knobs),
                rationale_summary=draft_descriptor.rationale_summary,
                estimated_risk=draft_descriptor.estimated_risk,
                parent_branch=self.accepted_branch,
                parent_trial_id=None,
                budget_used=draft_descriptor.budget_used,
                diff_path=draft_descriptor.diff_path,
            )
            accepted_scored = self.score_descriptor(accepted_descriptor)
        accepted = {
            "branch_name": self.accepted_branch,
            "trial_id": None,
            "source": "accepted_branch",
            "patch_category": None,
            "canonical_score": None,
            "posterior_mean": 0.0 if accepted_scored is None else accepted_scored["posterior_mean"],
            "posterior_std": 1.0 if accepted_scored is None else accepted_scored["posterior_std"],
            "acquisition_score": (suggestions[0]["acquisition_score"] - 0.01)
            if accepted_scored is None
            else accepted_scored["acquisition_score"],
            "similarity_to_request": None,
            "reason": "Use the current accepted branch as the clean baseline.",
        }
        ranked = [accepted, *suggestions]
        ranked.sort(key=lambda item: float(item["acquisition_score"]), reverse=True)
        return ranked[:count]

    def rank_patch_families(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        count = max(1, int(request.get("limit") or self.config.search.family_suggestion_count))
        base_request = dict(request)
        base_request.pop("limit", None)
        family_rows: list[dict[str, Any]] = []
        for category in sorted(PATCH_CATEGORIES):
            family_request = dict(base_request)
            family_request["patch_category"] = category
            if "operation_type" not in family_request:
                family_request["operation_type"] = "replace"
            if "estimated_risk" not in family_request:
                family_request["estimated_risk"] = 0.25
            scored = self.score_candidate_descriptor(family_request)
            evidence = [
                trial
                for trial in descriptor_trials(self.memory)
                if trial.descriptor and trial.descriptor.patch_category == category
            ]
            family_rows.append(
                {
                    "patch_category": category,
                    "acquisition_score": scored["acquisition_score"],
                    "posterior_mean": scored["posterior_mean"],
                    "posterior_std": scored["posterior_std"],
                    "trial_count": len(evidence),
                    "best_score": max((float(trial.canonical_score or 0.0) for trial in evidence), default=None),
                }
            )
        family_rows.sort(key=lambda item: float(item["acquisition_score"]), reverse=True)
        return family_rows[:count]

    def propose_numeric_knob_regions(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        count = max(1, int(request.get("limit") or self.config.search.knob_region_count))
        requested_category = str(request.get("patch_category") or "").strip().lower()
        pool = [
            trial
            for trial in descriptor_trials(self.memory)
            if trial.descriptor and (not requested_category or trial.descriptor.patch_category == requested_category)
        ]
        observations: dict[str, list[float]] = {}
        for trial in pool:
            assert trial.descriptor is not None
            for key, value in trial.descriptor.numeric_knobs.items():
                observations.setdefault(key, []).append(float(value))
        if not observations:
            requested_knobs = dict(request.get("numeric_knobs") or {})
            for key, value in requested_knobs.items():
                center = float(value)
                span = max(abs(center) * 0.2, 0.1)
                observations[str(key)] = [center - span, center, center + span]
        regions: list[dict[str, Any]] = []
        for name, values in observations.items():
            sorted_values = sorted(values)
            center = sum(sorted_values) / len(sorted_values)
            regions.append(
                {
                    "name": name,
                    "lower": sorted_values[0],
                    "upper": sorted_values[-1],
                    "center": center,
                    "count": len(sorted_values),
                }
            )
        regions.sort(key=lambda item: (-int(item["count"]), str(item["name"])))
        return regions[:count]
