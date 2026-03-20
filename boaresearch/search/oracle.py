from __future__ import annotations

from typing import Any, Optional

from ..schema import BoaConfig, DescriptorDraft, LessonRecord, PATCH_CATEGORIES, PatchDescriptor, TrialSummary
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


def build_lesson_id(trial_id: str, fix_index: int) -> str:
    return f"{trial_id}:lesson:{fix_index + 1}"


def _trim_text(text: Any, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _trim_string_list(values: list[str], *, limit: int, max_items: int) -> list[str]:
    trimmed: list[str] = []
    for raw in list(values or []):
        value = _trim_text(raw, limit)
        if value:
            trimmed.append(value)
        if len(trimmed) >= max_items:
            break
    return trimmed


def _trial_addressed_lesson_ids(trial: TrialSummary) -> list[str]:
    if trial.candidate and trial.candidate.addressed_lesson_ids:
        return list(trial.candidate.addressed_lesson_ids)
    if trial.candidate_plan and trial.candidate_plan.addressed_lesson_ids:
        return list(trial.candidate_plan.addressed_lesson_ids)
    return []


def trial_dataset_row(trial: TrialSummary) -> dict[str, Any]:
    descriptor = trial.descriptor
    candidate = trial.candidate
    candidate_plan = trial.candidate_plan
    return {
        "trial_id": trial.trial_id,
        "branch_name": trial.branch_name,
        "parent_branch": trial.parent_branch,
        "parent_trial_id": trial.parent_trial_id,
        "acceptance_status": trial.acceptance_status,
        "canonical_stage": trial.canonical_stage,
        "canonical_score": trial.canonical_score,
        "stage_scores": dict(trial.stage_scores),
        "patch_category": descriptor.patch_category if descriptor else None,
        "operation_type": descriptor.operation_type if descriptor else None,
        "estimated_risk": descriptor.estimated_risk if descriptor else None,
        "numeric_knobs": dict(descriptor.numeric_knobs) if descriptor else {},
        "touched_files": list(descriptor.touched_files) if descriptor else [],
        "target_symbols": list(descriptor.touched_symbols) if descriptor else [],
        "rationale_summary": candidate.rationale_summary if candidate else None,
        "hypothesis": candidate.hypothesis if candidate else None,
        "addressed_lesson_ids": _trial_addressed_lesson_ids(trial),
        "planned_parent_branch": candidate_plan.selected_parent_branch if candidate_plan else None,
        "planned_parent_trial_id": candidate_plan.selected_parent_trial_id if candidate_plan else None,
    }


class SearchOracleService:
    def __init__(self, *, config: BoaConfig, memory: list[TrialSummary], accepted_branch: str) -> None:
        self.config = config
        self.memory = [trial for trial in memory if trial.run_tag == config.run.tag]
        self.accepted_branch = accepted_branch

    def recent_trials(self, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
        rows = self.memory[: max(1, int(limit or self.config.search.max_history))]
        return [summary(trial) for trial in rows]

    def planning_trial_dataset(self, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
        rows = self.memory[: max(1, int(limit or self.config.search.max_history))]
        return [trial_dataset_row(trial) for trial in rows]

    def planning_lesson_memory(self, *, limit: int = 8) -> list[dict[str, Any]]:
        ordered = sorted(self.memory, key=lambda trial: ((trial.created_at or ""), trial.trial_id))
        lessons: list[tuple[str, LessonRecord]] = []
        for index, trial in enumerate(ordered):
            reflection = trial.reflection
            if reflection is None:
                continue
            suggested_fixes = _trim_string_list(reflection.suggested_fixes, limit=160, max_items=4)
            if not suggested_fixes:
                continue
            under_optimized = _trim_string_list(reflection.under_optimized, limit=80, max_items=4)
            evidence = _trim_string_list(reflection.evidence, limit=120, max_items=4)
            problem = _trim_text(reflection.primary_problem, 160)
            behavior = _trim_text(reflection.behavior_summary, 160)
            for fix_offset, suggested_fix in enumerate(suggested_fixes):
                lesson_id = build_lesson_id(trial.trial_id, fix_offset)
                later_trials = [
                    item
                    for item in ordered[index + 1 :]
                    if lesson_id in _trial_addressed_lesson_ids(item)
                ]
                has_success = any(item.acceptance_status == "completed_accepted" for item in later_trials)
                has_non_success = any(item.acceptance_status != "completed_accepted" for item in later_trials)
                if not later_trials:
                    status = "untested"
                elif has_success and has_non_success:
                    status = "mixed"
                elif has_success:
                    status = "succeeded"
                else:
                    status = "insufficient"
                latest_trial = later_trials[-1] if later_trials else trial
                lessons.append(
                    (
                        latest_trial.created_at or trial.created_at or "",
                        LessonRecord(
                            lesson_id=lesson_id,
                            source_trial_id=trial.trial_id,
                            source_stage=reflection.source_stage,
                            problem=problem,
                            behavior=behavior,
                            under_optimized=under_optimized,
                            suggested_fix=suggested_fix,
                            evidence=evidence,
                            status=status,
                            last_trial_id=latest_trial.trial_id,
                        ),
                    )
                )
        lessons.sort(key=lambda item: (item[0], item[1].lesson_id), reverse=True)
        return [
            {
                "lesson_id": record.lesson_id,
                "problem": record.problem,
                "behavior": record.behavior,
                "under_optimized": list(record.under_optimized),
                "suggested_fix": record.suggested_fix,
                "evidence": list(record.evidence),
                "status": record.status,
                "last_trial_id": record.last_trial_id,
            }
            for _sort_key, record in lessons[: max(1, int(limit))]
        ]

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
            # Some adapters may include a stale or non-empty selected_parent_trial_id
            # even when selecting the accepted branch. The accepted branch is always
            # a valid lineage root, so normalize trial_id away in this case.
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

    def planning_suggestion_report(self, *, limit: Optional[int] = None) -> dict[str, Any]:
        descriptor_trials_memory = descriptor_trials(self.memory)
        default_family = "misc"
        if descriptor_trials_memory:
            family_scores: dict[str, list[float]] = {}
            for trial in descriptor_trials_memory:
                assert trial.descriptor is not None
                family_scores.setdefault(trial.descriptor.patch_category, []).append(float(trial.canonical_score or 0.0))
            default_family = max(
                family_scores.items(),
                key=lambda item: ((sum(item[1]) / len(item[1])) if item[1] else float("-inf"), len(item[1])),
            )[0]
        baseline_request = {
            "patch_category": default_family,
            "operation_type": "replace",
            "estimated_risk": 0.25,
            "limit": max(1, int(limit or self.config.search.parent_suggestion_count)),
        }
        return {
            "report_type": "planning_bo_suggestions",
            "note": (
                "BOA suggestions are advisory only. The planning agent may use them, refine them, or ignore them "
                "when the repository evidence suggests a better move."
            ),
            "memory_trial_count": len(self.memory),
            "lineage_options": self.list_lineage_options(limit=limit),
            "parent_suggestions": self.suggest_parents(baseline_request),
            "patch_family_ranking": self.rank_patch_families(baseline_request),
            "numeric_knob_regions": self.propose_numeric_knob_regions(
                {
                    "patch_category": default_family,
                    "operation_type": "replace",
                    "estimated_risk": 0.25,
                    "limit": max(1, int(limit or self.config.search.knob_region_count)),
                }
            ),
        }
