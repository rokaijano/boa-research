from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boaresearch.descriptors import build_patch_descriptor
from boaresearch.schema import CandidateMetadata


class DescriptorTests(unittest.TestCase):
    def test_build_patch_descriptor_merges_symbols_and_knobs(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        diff_path = temp_dir / "patch.diff"
        diff_text = """diff --git a/src/train.py b/src/train.py
--- a/src/train.py
+++ b/src/train.py
@@ -1,3 +1,4 @@ def train_epoch
+def train_epoch(config):
+    learning_rate = 0.0002
"""
        diff_path.write_text(diff_text, encoding="utf-8")
        candidate = CandidateMetadata(
            hypothesis="Improve optimizer settings",
            rationale_summary="Tune the optimizer.",
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            target_symbols=["Trainer.train_epoch"],
            numeric_knobs={"weight_decay": 0.02},
        )
        descriptor = build_patch_descriptor(
            touched_files=["src/train.py"],
            diff_text=diff_text,
            candidate=candidate,
            parent_branch="boa/demo/accepted",
            parent_trial_id="demo-0001",
            budget_used="scout",
            diff_path=diff_path,
        )
        self.assertEqual(descriptor.touched_files, ["src/train.py"])
        self.assertIn("Trainer.train_epoch", descriptor.touched_symbols)
        self.assertIn("learning_rate", descriptor.numeric_knobs)
        self.assertEqual(descriptor.numeric_knobs["weight_decay"], 0.02)


if __name__ == "__main__":
    unittest.main()
