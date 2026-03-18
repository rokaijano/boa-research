from __future__ import annotations

import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from boaresearch.init import build_config_from_plan, default_selection_for_repo, detect_repo, merge_reviewed_plan, render_boa_md, render_config_text
from boaresearch.loader import load_config
from boaresearch.init.models import InitSetupSelection, RepoAnalysisProposal, ReviewedInitPlan


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

    def test_render_boa_md_includes_light_aggressiveness_guidance(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        selection = InitSetupSelection(repo_root=repo_root, agent_aggressiveness="light")
        analysis = RepoAnalysisProposal(
            train_command="python train.py",
            eval_command="python eval.py",
            primary_metric_name="accuracy",
            metric_direction="maximize",
            metric_source="regex",
            metric_path=None,
            metric_json_key=None,
            metric_pattern="accuracy=([0-9.]+)",
            editable_files=["src"],
            protected_files=[".boa"],
            optimization_surfaces=["src"],
            caveats=["Confirm metric extraction"],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = ReviewedInitPlan(
            repo_root=repo_root,
            selection=selection,
            analysis=analysis,
            editable_files=["src"],
            protected_files=[".boa", ".git"],
            train_command=analysis.train_command,
            eval_command=analysis.eval_command,
            primary_metric_name=analysis.primary_metric_name,
            metric_direction=analysis.metric_direction,
            metric_source=analysis.metric_source,
            metric_path=analysis.metric_path,
            metric_json_key=analysis.metric_json_key,
            metric_pattern=analysis.metric_pattern,
            boa_md=analysis.suggested_boa_md,
        )

        rendered = render_boa_md(plan)

        self.assertIn("Aggressiveness: `light`", rendered)
        self.assertIn("Make the smallest viable diff", rendered)

    def test_render_boa_md_includes_aggressive_guidance(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        selection = InitSetupSelection(repo_root=repo_root, agent_aggressiveness="aggressive")
        analysis = RepoAnalysisProposal(
            train_command="python train.py",
            eval_command="python eval.py",
            primary_metric_name="accuracy",
            metric_direction="maximize",
            metric_source="regex",
            metric_path=None,
            metric_json_key=None,
            metric_pattern="accuracy=([0-9.]+)",
            editable_files=["src"],
            protected_files=[".boa"],
            optimization_surfaces=["src"],
            caveats=["Confirm metric extraction"],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = ReviewedInitPlan(
            repo_root=repo_root,
            selection=selection,
            analysis=analysis,
            editable_files=["src"],
            protected_files=[".boa", ".git"],
            train_command=analysis.train_command,
            eval_command=analysis.eval_command,
            primary_metric_name=analysis.primary_metric_name,
            metric_direction=analysis.metric_direction,
            metric_source=analysis.metric_source,
            metric_path=analysis.metric_path,
            metric_json_key=analysis.metric_json_key,
            metric_pattern=analysis.metric_pattern,
            boa_md=analysis.suggested_boa_md,
        )

        rendered = render_boa_md(plan)

        self.assertIn("Aggressiveness: `aggressive`", rendered)
        self.assertIn("Broader rewrites are allowed", rendered)

    def test_update_mode_preserves_existing_advanced_settings(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo_root, check=True, capture_output=True)
        (repo_root / "boa.md").write_text("# BOA\n", encoding="utf-8")
        (repo_root / "boa.config").write_text(
            textwrap.dedent(
                """
                schema_version = 3

                [run]
                tag = "kept"
                max_trials = 4

                [agent]
                preset = "codex"
                runtime = "cli"
                command = "codex"
                extra_context_files = ["README.md"]

                [guardrails]
                allowed_paths = ["src"]
                protected_paths = [".boa"]

                [git_auth]
                token_env = "CUSTOM_GIT_TOKEN"
                fallback_token_env = "CUSTOM_FALLBACK"
                username = "token-user"

                [runner]
                mode = "local"

                [runner.local]

                [runner.ssh]
                host_alias = "train-box"
                repo_path = "~/repo"
                git_remote = "origin"
                runtime_root = ".boa/remote"

                [runner.scout]
                enabled = true
                commands = ["python train.py", "python eval.py"]
                timeout_seconds = 1200

                [runner.confirm]
                enabled = true
                commands = ["python train.py --confirm"]
                timeout_seconds = 2400

                [[metrics]]
                name = "accuracy"
                source = "json_file"
                path = "reports/metrics.json"
                json_key = "accuracy"

                [[metrics]]
                name = "latency"
                source = "regex"
                pattern = "latency=([0-9.]+)"

                [objective]
                primary_metric = "accuracy"
                direction = "maximize"
                threshold = 0.8

                [search]
                oracle = "bayesian_optimization"
                seed = 11
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (repo_root / "README.md").write_text("context\n", encoding="utf-8")
        detected = detect_repo(repo_root)
        selection = default_selection_for_repo(detected)
        analysis = RepoAnalysisProposal(
            train_command="python train.py",
            eval_command="python eval.py",
            primary_metric_name="accuracy",
            metric_direction="maximize",
            metric_source="json_file",
            metric_path="reports/metrics.json",
            metric_json_key="accuracy",
            metric_pattern=None,
            editable_files=["src"],
            protected_files=[".boa"],
            optimization_surfaces=["src"],
            caveats=[],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = merge_reviewed_plan(selection, analysis)
        config = build_config_from_plan(plan)
        self.assertEqual(config.run.tag, "kept")
        self.assertEqual(config.git_auth.token_env, "CUSTOM_GIT_TOKEN")
        self.assertTrue(config.runner.confirm.enabled)
        self.assertEqual(config.objective.threshold, 0.8)
        self.assertEqual(config.search.seed, 11)
        self.assertEqual([metric.name for metric in config.metrics], ["accuracy", "latency"])

        (repo_root / "boa.config").write_text(render_config_text(plan), encoding="utf-8")
        loaded = load_config(repo_root)
        self.assertEqual(loaded.run.tag, "kept")
        self.assertEqual(loaded.git_auth.token_env, "CUSTOM_GIT_TOKEN")
        self.assertTrue(loaded.runner.confirm.enabled)
        self.assertEqual(loaded.objective.threshold, 0.8)
        self.assertEqual(loaded.search.seed, 11)
        self.assertEqual([metric.name for metric in loaded.metrics], ["accuracy", "latency"])


if __name__ == "__main__":
    unittest.main()
