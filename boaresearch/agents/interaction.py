from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..schema import CandidateMetadata, CandidatePlan
from .base import extract_json_object, parse_candidate_dict, parse_candidate_plan_dict


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

    @staticmethod
    def persist_plan(*, plan_path: Path, plan: CandidatePlan) -> None:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def persist_candidate(*, candidate_path: Path, candidate: CandidateMetadata) -> None:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(json.dumps(asdict(candidate), indent=2, sort_keys=True) + "\n", encoding="utf-8")
