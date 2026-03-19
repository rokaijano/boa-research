from __future__ import annotations

import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from boaresearch.loader import load_config


class LoaderTests(unittest.TestCase):
    def _make_repo(self, *, mode: str = "local") -> Path:
        temp_dir = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=temp_dir, check=True, capture_output=True)
        (temp_dir / "boa.md").write_text("# BOA\n\nOnly edit src.\n", encoding="utf-8")
        ssh_block = textwrap.dedent(
            """
            [runner.ssh]
            host_alias = "train-box"
            repo_path = "~/repos/demo"
            git_remote = "origin"
            runtime_root = ".boa/remote"
            """
        ).strip()
        (temp_dir / "boa.config").write_text(
            textwrap.dedent(
                f"""
                schema_version = 3

                [run]
                tag = "demo"
                max_trials = 2

                [agent]
                preset = "codex"
                runtime = "cli"
                command = "codex"

                [guardrails]
                allowed_paths = ["src"]
                protected_paths = [".boa/protected"]

                [runner]
                mode = "{mode}"

                [runner.local]

                {ssh_block}

                [runner.scout]
                enabled = true
                commands = ["python train.py", "python eval.py"]
                timeout_seconds = 1200

                [runner.confirm]
                enabled = true
                commands = ["python train.py --confirm"]
                timeout_seconds = 1800

                [[metrics]]
                name = "accuracy"
                source = "json_file"
                path = "reports/metrics.json"
                json_key = "accuracy"

                [objective]
                primary_metric = "accuracy"
                direction = "maximize"

                [search]
                oracle = "bayesian_optimization"
                seed = 7
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return temp_dir

    def test_load_config_from_repo_root_local(self) -> None:
        repo = self._make_repo(mode="local")
        config = load_config(repo)
        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.agent.runtime, "cli")
        self.assertEqual(config.runner.mode, "local")
        self.assertEqual(config.run.accepted_branch, "boa/demo/accepted")
        self.assertEqual(config.guardrails.allowed_paths, ["src"])
        self.assertEqual(config.guardrails.protected_paths, [".boa/protected"])
        self.assertEqual(config.search.oracle, "bayesian_optimization")
        self.assertTrue(str(config.run.worktree_path).replace("\\", "/").endswith("/.boa/worktree/demo"))

    def test_hidden_paths_are_not_stripped_during_normalization(self) -> None:
        repo = self._make_repo(mode="local")
        (repo / "boa.config").write_text(
            (repo / "boa.config").read_text(encoding="utf-8").replace(
                'protected_paths = [".boa/protected"]',
                'protected_paths = ["./.boa", "./.gitignore"]',
            ),
            encoding="utf-8",
        )
        config = load_config(repo)
        self.assertEqual(config.guardrails.protected_paths, [".boa/protected", ".gitignore"])

    def test_load_config_from_repo_root_ssh(self) -> None:
        repo = self._make_repo(mode="ssh")
        config = load_config(repo)
        self.assertEqual(config.runner.mode, "ssh")
        self.assertEqual(config.runner.ssh.host_alias, "train-box")

    def test_missing_boa_md_raises(self) -> None:
        repo = self._make_repo()
        (repo / "boa.md").unlink()
        with self.assertRaises(FileNotFoundError):
            load_config(repo)

    def test_old_config_is_rejected(self) -> None:
        repo = self._make_repo()
        (repo / "boa.config").write_text("[run]\ntag = \"demo\"\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(repo)

    def test_policy_field_is_rejected_in_v3(self) -> None:
        repo = self._make_repo()
        (repo / "boa.config").write_text(
            (repo / "boa.config").read_text(encoding="utf-8").replace(
                'oracle = "bayesian_optimization"',
                'policy = "bayesian_optimization"',
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_config(repo)


if __name__ == "__main__":
    unittest.main()
