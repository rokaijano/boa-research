from __future__ import annotations

from .oracle import SearchOracleService
from .trace import SearchTraceRecorder


class SearchToolbox:
    def __init__(self, *, oracle: SearchOracleService, recorder: SearchTraceRecorder) -> None:
        self.oracle = oracle
        self.recorder = recorder

    def recent_trials(self, request: dict[str, object]) -> dict[str, object]:
        response = {"trials": self.oracle.recent_trials(limit=request.get("limit"))}
        return self.recorder.record(tool_name="recent_trials", request=request, response=response)

    def list_lineage_options(self, request: dict[str, object]) -> dict[str, object]:
        response = {"options": self.oracle.list_lineage_options(limit=request.get("limit"))}
        return self.recorder.record(tool_name="list_lineage_options", request=request, response=response)

    def suggest_parents(self, request: dict[str, object]) -> dict[str, object]:
        response = {"suggestions": self.oracle.suggest_parents(request)}
        return self.recorder.record(tool_name="suggest_parents", request=request, response=response)

    def score_candidate_descriptor(self, request: dict[str, object]) -> dict[str, object]:
        response = self.oracle.score_candidate_descriptor(request)
        return self.recorder.record(tool_name="score_candidate_descriptor", request=request, response=response)

    def rank_patch_families(self, request: dict[str, object]) -> dict[str, object]:
        response = {"families": self.oracle.rank_patch_families(request)}
        return self.recorder.record(tool_name="rank_patch_families", request=request, response=response)

    def propose_numeric_knob_regions(self, request: dict[str, object]) -> dict[str, object]:
        response = {"regions": self.oracle.propose_numeric_knob_regions(request)}
        return self.recorder.record(tool_name="propose_numeric_knob_regions", request=request, response=response)
