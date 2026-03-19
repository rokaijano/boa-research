from __future__ import annotations

import io
import unittest

from boaresearch.runtime.observer import RunEvent
from boaresearch.runtime.monitor import _RICH_AVAILABLE, RichRunObserver, build_run_observer


class MonitorTests(unittest.TestCase):
    def test_build_run_observer_plain_mode(self) -> None:
        stream = io.StringIO()
        observer = build_run_observer(stream=stream, interactive=False)
        self.assertEqual(observer.__class__.__name__, "PlainRunObserver")

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_initializes_before_first_render(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)
        self.assertIsNone(observer.current_trial_id)
        self.assertEqual(observer.current_status, "idle")

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_tracks_trial_metadata_for_lineage(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="execution_started",
                message="Preparing execution worktree from boa/default/trial/default-0001",
                trial_id="demo-0002",
                phase="execution",
                metadata={
                    "branch_name": "boa/default/trial/demo-0002",
                    "parent_branch": "boa/default/trial/default-0001",
                    "parent_trial_id": "default-0001",
                },
            )
        )
        observer.emit(
            RunEvent(
                kind="stage_completed",
                message="Stage 'scout' completed | score=0.980000 | accuracy=0.980000",
                trial_id="demo-0002",
                phase="evaluation",
                stage_name="scout",
                metadata={
                    "score": 0.98,
                    "primary_metric": 0.98,
                    "branch_name": "boa/default/trial/demo-0002",
                },
            )
        )
        observer.emit(
            RunEvent(
                kind="trial_completed",
                message="Trial demo-0002 finished with status completed_accepted",
                trial_id="demo-0002",
                metadata={
                    "acceptance_status": "completed_accepted",
                    "branch_name": "boa/default/trial/demo-0002",
                    "parent_branch": "boa/default/trial/default-0001",
                    "parent_trial_id": "default-0001",
                    "canonical_stage": "scout",
                    "canonical_score": 0.98,
                    "primary_metric": 0.98,
                },
            )
        )

        record = observer.trials["demo-0002"]
        self.assertEqual(record["parent_trial_id"], "default-0001")
        self.assertEqual(record["acceptance_status"], "completed_accepted")
        self.assertEqual(record["primary_metric"], 0.98)

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_filters_usage_noise_from_agent_output(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="Total usage est:        0 Premium requests",
                trial_id="demo-0001",
                phase="planning",
                source="agent.stderr",
            )
        )

        self.assertEqual(len(observer.activity_lines), 0)
        self.assertEqual(len(observer.error_lines), 0)

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_keeps_high_signal_agent_and_error_output(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="● Edit model.py +9 -7",
                trial_id="demo-0001",
                phase="planning",
                source="agent.stdout",
            )
        )
        observer.emit(
            RunEvent(
                kind="process_output",
                message="Invalid shell ID: 1. Please supply a valid shell ID to read output from.",
                trial_id="demo-0001",
                phase="execution",
                source="agent.stderr",
            )
        )

        self.assertEqual(len(observer.activity_lines), 1)
        self.assertIn("Edit model.py", observer.activity_lines[0])
        self.assertEqual(len(observer.error_lines), 1)
        self.assertIn("Invalid shell ID", observer.error_lines[0])

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_filters_percent_only_terminal_output(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="15.2%",
                trial_id="demo-0001",
                phase="evaluation",
                stage_name="scout",
                source="stage.stderr",
            )
        )
        observer.emit(
            RunEvent(
                kind="process_output",
                message="epoch=1 loss=0.42",
                trial_id="demo-0001",
                phase="execution",
                stage_name="scout",
                source="stage.stdout",
            )
        )

        self.assertEqual(len(observer.activity_lines), 1)
        self.assertIn("epoch=1 loss=0.42", observer.activity_lines[0])

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_records_structured_events_as_activity(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="planning_result",
                message="Plan ready: architecture/replace | parent=boa/default/accepted | risk=0.25",
                trial_id="demo-0001",
                phase="planning",
            )
        )

        self.assertEqual(len(observer.activity_lines), 1)
        self.assertIn("Plan ready", observer.activity_lines[0])

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_uses_transient_status_for_process_waiting(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_waiting",
                message="Agent process still running...",
                trial_id="demo-0001",
                phase="planning",
                source="agent.stdout",
            )
        )

        self.assertEqual(len(observer.activity_lines), 0)
        self.assertEqual(observer.transient_status, "Agent process still running...")


if __name__ == "__main__":
    unittest.main()
