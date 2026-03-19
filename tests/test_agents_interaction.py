from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from boaresearch.agents.interaction import BoaInteractionLayer


class BoaInteractionLayerTests(unittest.TestCase):
    def test_parse_plan_output_prefers_file_over_stdout(self) -> None:
        root = Path(tempfile.mkdtemp())
        layer = BoaInteractionLayer()
        plan_path = root / "plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "hypothesis": "h-file",
                    "rationale_summary": "r",
                    "selected_parent_branch": "boa/demo/accepted",
                    "patch_category": "optimizer",
                    "operation_type": "replace",
                    "estimated_risk": 0.2,
                    "informed_by_call_ids": ["boa-call-1"],
                }
            ),
            encoding="utf-8",
        )
        plan = layer.parse_plan_output(
            plan_path=plan_path,
            stdout=json.dumps(
                {
                    "hypothesis": "h-stdout",
                    "rationale_summary": "r",
                    "selected_parent_branch": "boa/demo/trial/other",
                    "patch_category": "optimizer",
                    "operation_type": "replace",
                    "estimated_risk": 0.2,
                    "informed_by_call_ids": ["boa-call-2"],
                }
            ),
        )
        self.assertEqual(plan.hypothesis, "h-file")

    def test_parse_candidate_output_uses_stdout_fallback(self) -> None:
        root = Path(tempfile.mkdtemp())
        layer = BoaInteractionLayer()
        candidate = layer.parse_candidate_output(
            candidate_path=root / "missing.json",
            stdout=json.dumps(
                {
                    "hypothesis": "h-stdout",
                    "rationale_summary": "r",
                    "patch_category": "optimizer",
                    "operation_type": "replace",
                    "estimated_risk": 0.2,
                    "informed_by_call_ids": ["boa-call-2"],
                }
            ),
        )
        self.assertEqual(candidate.hypothesis, "h-stdout")

    def test_persist_helpers_write_valid_json(self) -> None:
        root = Path(tempfile.mkdtemp())
        layer = BoaInteractionLayer()
        plan = layer.parse_plan_payload(
            {
                "hypothesis": "h",
                "rationale_summary": "r",
                "selected_parent_branch": "boa/demo/accepted",
                "patch_category": "optimizer",
                "operation_type": "replace",
                "estimated_risk": 0.2,
                "informed_by_call_ids": ["boa-call-1"],
            }
        )
        candidate = layer.parse_candidate_payload(
            {
                "hypothesis": "h",
                "rationale_summary": "r",
                "patch_category": "optimizer",
                "operation_type": "replace",
                "estimated_risk": 0.2,
                "informed_by_call_ids": ["boa-call-2"],
            }
        )

        plan_path = root / "out" / "plan.json"
        candidate_path = root / "out" / "candidate.json"
        layer.persist_plan(plan_path=plan_path, plan=plan)
        layer.persist_candidate(candidate_path=candidate_path, candidate=candidate)

        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual(plan_payload["selected_parent_branch"], "boa/demo/accepted")
        self.assertEqual(candidate_payload["patch_category"], "optimizer")


if __name__ == "__main__":
    unittest.main()
