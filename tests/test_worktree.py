from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boaresearch.runtime.worktree import WorktreeManager


class WorktreeManagerTests(unittest.TestCase):
    def test_cleanup_scratch_artifacts_removes_root_temp_entries(self) -> None:
        repo = Path(tempfile.mkdtemp())
        worktree = repo / "worktree"
        worktree.mkdir()
        scratch_dir = worktree / "_tmp_artifacts"
        scratch_dir.mkdir()
        (scratch_dir / "artifact.txt").write_text("scratch", encoding="utf-8")
        scratch_file = worktree / "tmp_notes.txt"
        scratch_file.write_text("scratch", encoding="utf-8")
        manager = WorktreeManager(repo_root=repo, worktree_path=worktree, accepted_branch="boa/demo/accepted")

        cleaned = manager.cleanup_scratch_artifacts()

        self.assertEqual(sorted(cleaned), ["_tmp_artifacts", "tmp_notes.txt"])
        self.assertFalse(scratch_dir.exists())
        self.assertFalse(scratch_file.exists())


if __name__ == "__main__":
    unittest.main()
