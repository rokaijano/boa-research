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
                    "addressed_lesson_ids": [],
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
                    "addressed_lesson_ids": [],
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
                    "addressed_lesson_ids": [],
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
                "addressed_lesson_ids": [],
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
                "addressed_lesson_ids": [],
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

    def test_parse_reflection_output_uses_stdout_fallback(self) -> None:
        root = Path(tempfile.mkdtemp())
        layer = BoaInteractionLayer()
        reflection = layer.parse_reflection_output(
            reflection_path=root / "missing.json",
            stdout=json.dumps(
                {
                    "source_stage": "scout",
                    "source_commands": ["python train.py"],
                    "behavior_summary": "Validation stalled after early gains.",
                    "primary_problem": "Generalization plateaued.",
                    "under_optimized": ["regularization"],
                    "suggested_fixes": ["Increase weight decay slightly."],
                    "evidence": ["train loss fell while val accuracy stalled"],
                    "outcome": "Attempted patch was insufficient.",
                }
            ),
        )
        self.assertEqual(reflection.source_stage, "scout")

    def test_persist_reflection_writes_valid_json(self) -> None:
        root = Path(tempfile.mkdtemp())
        layer = BoaInteractionLayer()
        reflection = layer.parse_reflection_payload(
            {
                "source_stage": "scout",
                "source_commands": ["python train.py"],
                "behavior_summary": "Validation stalled after early gains.",
                "primary_problem": "Generalization plateaued.",
                "under_optimized": ["regularization"],
                "suggested_fixes": ["Increase weight decay slightly."],
                "evidence": ["train loss fell while val accuracy stalled"],
                "outcome": "Attempted patch was insufficient.",
            }
        )
        reflection_path = root / "out" / "reflection.json"
        layer.persist_reflection(reflection_path=reflection_path, reflection=reflection)
        payload = json.loads(reflection_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["source_stage"], "scout")


if __name__ == "__main__":
    unittest.main()
