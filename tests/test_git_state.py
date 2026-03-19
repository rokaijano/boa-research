from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from boaresearch import git_state


class GitStateTests(unittest.TestCase):
    def test_current_branch_handles_unborn_head(self) -> None:
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)

        self.assertEqual(git_state.current_branch(repo), "main")
        self.assertFalse(git_state.has_commits(repo))


if __name__ == "__main__":
    unittest.main()
