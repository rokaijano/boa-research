from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..schema import CandidateMetadata, CandidatePlan, TrialReflection
from .base import extract_json_object, parse_candidate_dict, parse_candidate_plan_dict, parse_trial_reflection_dict


class BoaInteractionLayer:
    """Shared BOA interaction contract for agent adapters.

    This layer centralizes how adapters:
    - parse candidate plan/candidate metadata from file-or-stdout fallbacks
    - validate via schema normalizers
    - persist normalized outputs to BOA-owned paths
    """

    @staticmethod
    def parse_plan_payload(payload: dict[str, Any]) -> CandidatePlan:
        return parse_candidate_plan_dict(payload)

    @staticmethod
    def parse_candidate_payload(payload: dict[str, Any]) -> CandidateMetadata:
        return parse_candidate_dict(payload)

    @staticmethod
    def parse_reflection_payload(payload: dict[str, Any]) -> TrialReflection:
        return parse_trial_reflection_dict(payload)

    def parse_plan_output(self, *, plan_path: Path, stdout: str) -> CandidatePlan:
        if plan_path.exists():
            data = extract_json_object(plan_path.read_text(encoding="utf-8"))
            return self.parse_plan_payload(data)
        data = extract_json_object(stdout)
        return self.parse_plan_payload(data)

    def parse_candidate_output(self, *, candidate_path: Path, stdout: str) -> CandidateMetadata:
        if candidate_path.exists():
            data = extract_json_object(candidate_path.read_text(encoding="utf-8"))
            return self.parse_candidate_payload(data)
        data = extract_json_object(stdout)
        return self.parse_candidate_payload(data)

    def parse_reflection_output(self, *, reflection_path: Path, stdout: str) -> TrialReflection:
        preferred_keys = {
            "source_stage",
            "source_commands",
            "behavior_summary",
            "primary_problem",
            "under_optimized",
            "suggested_fixes",
            "evidence",
            "outcome",
        }
        if reflection_path.exists():
            data = extract_json_object(reflection_path.read_text(encoding="utf-8"), preferred_keys=preferred_keys)
            return self.parse_reflection_payload(data)
        data = extract_json_object(stdout, preferred_keys=preferred_keys)
        return self.parse_reflection_payload(data)

    @staticmethod
    def persist_plan(*, plan_path: Path, plan: CandidatePlan) -> None:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def persist_candidate(*, candidate_path: Path, candidate: CandidateMetadata) -> None:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(json.dumps(asdict(candidate), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def persist_reflection(*, reflection_path: Path, reflection: TrialReflection) -> None:
        reflection_path.parent.mkdir(parents=True, exist_ok=True)
        reflection_path.write_text(json.dumps(asdict(reflection), indent=2, sort_keys=True) + "\n", encoding="utf-8")
