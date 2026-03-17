from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from boaresearch.init_services import detect_repo


class InitServicesTests(unittest.TestCase):
    def test_detect_repo_rejects_non_git_directory(self) -> None:
        path = Path(tempfile.mkdtemp())
        with self.assertRaises(ValueError):
            detect_repo(path)

    def test_detect_repo_resolves_git_root_from_subdirectory(self) -> None:
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        nested = repo / "src" / "inner"
        nested.mkdir(parents=True)
        detected = detect_repo(nested)
        self.assertEqual(detected.repo_root, repo.resolve())


if __name__ == "__main__":
    unittest.main()
