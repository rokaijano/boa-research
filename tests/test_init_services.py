from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from boaresearch.init import build_config_from_plan, default_repo_analysis, default_selection_for_repo, detect_repo, merge_reviewed_plan, render_boa_md, render_config_text
from boaresearch.init.services import _run_cli_analysis
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
            protected_files=[".boa/protected"],
            optimization_surfaces=["src"],
            caveats=["Confirm metric extraction"],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = ReviewedInitPlan(
            repo_root=repo_root,
            selection=selection,
            analysis=analysis,
            editable_files=["src"],
            protected_files=[".boa/protected", ".git"],
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
            protected_files=[".boa/protected"],
            optimization_surfaces=["src"],
            caveats=["Confirm metric extraction"],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = ReviewedInitPlan(
            repo_root=repo_root,
            selection=selection,
            analysis=analysis,
            editable_files=["src"],
            protected_files=[".boa/protected", ".git"],
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
                protected_paths = [".boa/protected"]

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
            protected_files=[".boa/protected"],
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

    def test_default_repo_analysis_protects_eval_and_metric_outputs(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        (repo_root / "src").mkdir()
        (repo_root / "data").mkdir()
        (repo_root / "tests").mkdir()
        (repo_root / "reports").mkdir()
        (repo_root / "README.md").write_text("# demo\n", encoding="utf-8")
        (repo_root / "train.py").write_text("print('train')\n", encoding="utf-8")
        (repo_root / "eval.py").write_text("print('eval')\n", encoding="utf-8")
        (repo_root / "data" / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
        (repo_root / "reports" / "metrics.json").write_text('{"accuracy": 1.0}\n', encoding="utf-8")

        analysis = default_repo_analysis(repo_root)

        self.assertEqual(analysis.editable_files, ["src"])
        self.assertIn(".boa/protected", analysis.protected_files)
        self.assertIn(".git", analysis.protected_files)
        self.assertIn("eval.py", analysis.protected_files)
        self.assertIn("data", analysis.protected_files)
        self.assertIn("reports/metrics.json", analysis.protected_files)
        self.assertNotIn(".gitignore", analysis.protected_files)
        self.assertNotIn("train.py", analysis.protected_files)
        self.assertNotIn("tests", analysis.protected_files)
        self.assertNotIn("README.md", analysis.protected_files)

    def test_merge_reviewed_plan_moves_sensitive_paths_to_protected(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        (repo_root / "src").mkdir()
        selection = InitSetupSelection(repo_root=repo_root)
        analysis = RepoAnalysisProposal(
            train_command="python train.py",
            eval_command="python eval.py",
            primary_metric_name="accuracy",
            metric_direction="maximize",
            metric_source="json_file",
            metric_path="reports/metrics.json",
            metric_json_key="accuracy",
            metric_pattern=None,
            editable_files=["src", "eval.py", "data", "train.py", "tests", "README.md", ".boa/protected"],
            protected_files=[],
            optimization_surfaces=["src"],
            caveats=[],
            suggested_boa_md="# BOA Repo Contract",
        )

        plan = merge_reviewed_plan(selection, analysis)

        self.assertEqual(plan.editable_files, ["src", "train.py", "tests", "README.md"])
        self.assertIn(".boa/protected", plan.protected_files)
        self.assertIn(".git", plan.protected_files)
        self.assertIn("eval.py", plan.protected_files)
        self.assertIn("data", plan.protected_files)
        self.assertIn("reports/metrics.json", plan.protected_files)
        self.assertNotIn("README.md", plan.protected_files)
        self.assertNotIn("tests", plan.protected_files)
        self.assertNotIn("train.py", plan.protected_files)

    def test_run_cli_analysis_resolves_windows_bat_command(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="copilot", agent_runtime="cli", agent_command="copilot")
        completed = subprocess.CompletedProcess(args=["copilot"], returncode=0, stdout='{"train_command":"python train.py","eval_command":"python eval.py","primary_metric_name":"accuracy","metric_direction":"maximize","metric_source":"regex","metric_path":null,"metric_json_key":null,"metric_pattern":"accuracy=([0-9.]+)","editable_files":["src"],"protected_files":[".boa/protected"],"optimization_surfaces":["src"],"caveats":[],"suggested_boa_md":"# BOA"}', stderr="")

        with patch("boaresearch.init.services.shutil.which", return_value=r"C:\\tools\\copilot.BAT"):
            with patch("boaresearch.init.services.subprocess.run", return_value=completed) as run_mock:
                analysis = _run_cli_analysis(selection, "prompt", repo_root)

        self.assertEqual(analysis.train_command, "python train.py")
        self.assertEqual(run_mock.call_args.args[0][0], r"C:\\tools\\copilot.BAT")

    def test_run_codex_cli_analysis_closes_schema_file_before_launch(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_runtime="cli", agent_command="codex")
        schema_file_name = str(repo_root / "schema.json")
        output_file_name = str(repo_root / "output.txt")

        class DummyTempFile:
            def __init__(self, name: str) -> None:
                self.name = name
                self.closed = False
                self._buffer = ""

            def write(self, data: str) -> int:
                self._buffer += data
                return len(data)

            def flush(self) -> None:
                observed[f"{self.name}:flushed"] = True

            def close(self) -> None:
                if not self.closed:
                    self.closed = True

        schema_file = DummyTempFile(schema_file_name)
        output_file = DummyTempFile(output_file_name)
        observed: dict[str, object] = {}
        analysis_payload = '{"train_command":"python train.py","eval_command":"python eval.py","primary_metric_name":"accuracy","metric_direction":"maximize","metric_source":"regex","metric_path":null,"metric_json_key":null,"metric_pattern":"accuracy=([0-9.]+)","editable_files":["src"],"protected_files":[".boa/protected"],"optimization_surfaces":["src"],"caveats":[],"suggested_boa_md":"# BOA"}'

        def fake_named_temporary_file(*args, **kwargs):
            suffix = kwargs.get("suffix")
            if suffix == ".json":
                return schema_file
            if suffix == ".txt":
                return output_file
            raise AssertionError(f"Unexpected suffix: {suffix}")

        def fake_run(command, **kwargs):
            observed["command"] = command
            observed["schema_closed"] = schema_file.closed
            observed["output_closed"] = output_file.closed
            observed["schema_buffer"] = schema_file._buffer
            observed["output_buffer"] = output_file._buffer
            observed["analysis_payload_written"] = True
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        with patch("boaresearch.init.services.tempfile.NamedTemporaryFile", side_effect=fake_named_temporary_file):
            with patch("boaresearch.init.services.shutil.which", return_value=r"C:\\tools\\codex.exe"):
                with patch("boaresearch.init.services.subprocess.run", side_effect=fake_run):
                    with patch(
                        "boaresearch.init.services.parse_repo_analysis",
                        return_value=RepoAnalysisProposal(
                            train_command="python train.py",
                            eval_command="python eval.py",
                            primary_metric_name="accuracy",
                            metric_direction="maximize",
                            metric_source="regex",
                            metric_path=None,
                            metric_json_key=None,
                            metric_pattern="accuracy=([0-9.]+)",
                            editable_files=["src"],
                            protected_files=[".boa/protected"],
                            optimization_surfaces=["src"],
                            caveats=[],
                            suggested_boa_md="# BOA",
                        ),
                    ):
                        analysis = _run_cli_analysis(selection, "prompt", repo_root)
        self.assertEqual(analysis.train_command, "python train.py")
        self.assertTrue(observed["schema_closed"])
        self.assertTrue(observed["output_closed"])
        self.assertIn('"train_command"', str(observed["schema_buffer"]))

    def test_run_codex_cli_analysis_sends_prompt_via_stdin(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_runtime="cli", agent_command="codex")
        prompt = "x" * 10000
        observed: dict[str, object] = {}
        analysis_payload = '{"train_command":"python train.py","eval_command":"python eval.py","primary_metric_name":"accuracy","metric_direction":"maximize","metric_source":"regex","metric_path":null,"metric_json_key":null,"metric_pattern":"accuracy=([0-9.]+)","editable_files":["src"],"protected_files":[".boa/protected"],"optimization_surfaces":["src"],"caveats":[],"suggested_boa_md":"# BOA"}'

        def fake_run(command, **kwargs):
            observed["command"] = command
            observed["input"] = kwargs.get("input")
            observed["schema_closed"] = schema_file.closed
            observed["output_closed"] = output_file.closed
            observed["schema_buffer"] = schema_file._buffer
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        class DummyTempFile:
            def __init__(self, name: str) -> None:
                self.name = name
                self._buffer = ""
                self.closed = False

            def write(self, data: str) -> int:
                self._buffer += data
                return len(data)

            def flush(self) -> None:
                observed[f"{self.name}:flushed"] = True

            def close(self) -> None:
                if not self.closed:
                    self.closed = True

        schema_file = DummyTempFile(str(repo_root / "schema.json"))
        output_file = DummyTempFile(str(repo_root / "output.txt"))

        def fake_named_temporary_file(*args, **kwargs):
            suffix = kwargs.get("suffix")
            if suffix == ".json":
                return schema_file
            if suffix == ".txt":
                return output_file
            raise AssertionError(f"Unexpected suffix: {suffix}")

        with patch("boaresearch.init.services.tempfile.NamedTemporaryFile", side_effect=fake_named_temporary_file):
            with patch("boaresearch.init.services.shutil.which", return_value=r"C:\\tools\\codex.exe"):
                with patch("boaresearch.init.services.subprocess.run", side_effect=fake_run):
                    with patch(
                        "boaresearch.init.services.parse_repo_analysis",
                        return_value=RepoAnalysisProposal(
                            train_command="python train.py",
                            eval_command="python eval.py",
                            primary_metric_name="accuracy",
                            metric_direction="maximize",
                            metric_source="regex",
                            metric_path=None,
                            metric_json_key=None,
                            metric_pattern="accuracy=([0-9.]+)",
                            editable_files=["src"],
                            protected_files=[".boa/protected"],
                            optimization_surfaces=["src"],
                            caveats=[],
                            suggested_boa_md="# BOA",
                        ),
                    ):
                        analysis = _run_cli_analysis(selection, prompt, repo_root)

        self.assertEqual(analysis.train_command, "python train.py")
        self.assertEqual(observed["input"], prompt)
        self.assertTrue(observed["schema_closed"])
        self.assertTrue(observed["output_closed"])
        command = observed["command"]
        self.assertIsInstance(command, list)
        self.assertIn("-", command)
        self.assertNotIn(prompt, " ".join(command))


if __name__ == "__main__":
    unittest.main()
