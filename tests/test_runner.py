from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from boaresearch.init import (
    default_repo_analysis,
    default_selection_for_repo,
    detect_repo,
    merge_reviewed_plan,
    validate_written_setup,
    write_contract_files,
)
from boaresearch.loader import load_config
from boaresearch.runtime.observer import RunEvent
from boaresearch.runner import LocalTrialRunner, SSHTrialRunner
from boaresearch.schema import LocalRunnerConfig, MetricConfig, RunnerStageConfig, SSHRunnerConfig


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class RunnerTests(unittest.TestCase):
    def test_local_runner_executes_and_extracts_metrics(self) -> None:
        repo = Path(tempfile.mkdtemp())
        (repo / "reports").mkdir()
        runner = LocalTrialRunner(LocalRunnerConfig())
        execution = runner.run_stage(
            trial_id="demo-0001",
            branch_name="boa/demo/trial/demo-0001",
            worktree_path=repo,
            stage_name="scout",
            stage=RunnerStageConfig(
                commands=[
                    "python3 -c \"from pathlib import Path; Path('reports/metrics.json').write_text('{\\\"accuracy\\\": 0.91}')\""
                ],
                timeout_seconds=30,
            ),
            metrics=[MetricConfig(name="accuracy", source="json_file", path="reports/metrics.json", json_key="accuracy")],
            artifact_dir=repo / ".artifacts",
        )
        self.assertEqual(execution.status, "succeeded")
        self.assertAlmostEqual(execution.metrics["accuracy"], 0.91)

    def test_ssh_runner_script_contains_timeout_and_metric_capture(self) -> None:
        runner = SSHTrialRunner(SSHRunnerConfig(host_alias="train-box", repo_path="~/repo", git_remote="origin"))
        script, remote_stage_dir = runner._build_stage_script(  # noqa: SLF001
            branch_name="boa/demo/trial/demo-0001",
            stage_name="scout",
            trial_id="demo-0001",
            stage=RunnerStageConfig(commands=["python train.py"], timeout_seconds=1200),
            metric_paths=["reports/metrics.json"],
        )
        self.assertIn("timeout 1200", script)
        self.assertIn("reports/metrics.json", script)
        self.assertEqual(remote_stage_dir, ".boa/remote/demo-0001/scout")

    def test_local_runner_streams_output_to_observer(self) -> None:
        repo = Path(tempfile.mkdtemp())
        observer = RecordingObserver()
        runner = LocalTrialRunner(LocalRunnerConfig(), observer=observer)

        execution = runner.run_stage(
            trial_id="demo-0001",
            branch_name="boa/demo/trial/demo-0001",
            worktree_path=repo,
            stage_name="scout",
            stage=RunnerStageConfig(commands=['python3 -c "print(\'runner-live\'); print(\'accuracy=0.77\')"'], timeout_seconds=30),
            metrics=[MetricConfig(name="accuracy", source="regex", pattern=r"accuracy=([0-9.]+)")],
            artifact_dir=repo / ".artifacts",
        )

        self.assertEqual(execution.status, "succeeded")
        self.assertAlmostEqual(execution.metrics["accuracy"], 0.77)
        self.assertTrue(any(event.kind == "process_output" and event.message == "runner-live" for event in observer.events))

    def test_init_written_config_is_loadable_and_validatable(self) -> None:
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
        (repo / "train.py").write_text("print('train')\n", encoding="utf-8")
        (repo / "eval.py").write_text("print('accuracy=0.5')\n", encoding="utf-8")
        detected = detect_repo(repo)
        selection = default_selection_for_repo(detected)
        selection.agent_command = "python3"
        selection.agent_preset = "custom"
        selection.agent_runtime = "cli"
        analysis = default_repo_analysis(repo)
        plan = merge_reviewed_plan(selection, analysis)
        write_contract_files(plan)
        config = load_config(repo)
        self.assertEqual(config.schema_version, 3)
        report = validate_written_setup(plan)
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
