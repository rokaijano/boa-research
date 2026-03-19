from __future__ import annotations

import io
import unittest

from boaresearch.runtime.observer import RunEvent
from boaresearch.init.banner import render_banner
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
    def test_rich_run_observer_renders_boa_banner_panel(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        rendered = observer._render()  # noqa: SLF001
        banner_panel = next(iter(rendered.renderables))
        banner_text = render_banner(width=120, height=24, allow_unicode=True)

        self.assertEqual(getattr(banner_panel, "title", None), "Bayesian Optimized Agents")
        self.assertIn("Bayesian Optimized Agents", banner_text)
        self.assertIn("\x1b[38;5;", banner_text)

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_routes_agent_and_terminal_output_separately(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="loss=0.42",
                trial_id="demo-0001",
                phase="evaluation",
                stage_name="scout",
                source="stage.stdout",
            )
        )
        observer.emit(
            RunEvent(
                kind="agent_prompt_sent",
                message="Sending planning prompt to codex",
                trial_id="demo-0001",
                phase="planning",
            )
        )

        self.assertEqual(list(observer.terminal_lines)[-1].endswith("loss=0.42"), True)
        self.assertIn("[agent]", observer.agent_lines[-1])
        self.assertTrue(observer.agent_lines[-1].endswith("Sending planning prompt to codex"))

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_routes_agent_process_output_to_agent_dialog(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="Work inside `src/mnist_demo/` unless a change outside that package is clearly necessary.",
                trial_id="demo-0001",
                phase="planning",
                source="agent.stdout",
            )
        )

        self.assertEqual(len(observer.agent_lines), 1)
        self.assertIn("[agent]", observer.agent_lines[0])
        self.assertTrue(observer.agent_lines[0].endswith("Work inside `src/mnist_demo/` unless a change outside that package is clearly necessary."))

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_routes_execution_agent_process_output_to_agent_dialog(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="epoch=1 loss=0.42",
                trial_id="demo-0001",
                phase="execution",
                stage_name="scout",
                source="agent.stdout",
            )
        )

        self.assertEqual(len(observer.agent_lines), 1)
        self.assertIn("[term]", observer.agent_lines[0])
        self.assertTrue(observer.agent_lines[0].endswith("epoch=1 loss=0.42"))

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_converts_agent_runtime_messages_to_tagged_dialog(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="agent_command_started",
                message="Waiting for codex planning response",
                trial_id="demo-0001",
                phase="planning",
            )
        )

        self.assertEqual(len(observer.agent_lines), 1)
        self.assertIn("[agent]", observer.agent_lines[0])
        self.assertTrue(observer.agent_lines[0].endswith("Waiting for codex planning response"))

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_agent_dialog_is_static_events_only(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="process_output",
                message="mcp startup: no servers",
                trial_id="demo-0001",
                phase="planning",
                source="agent.stdout",
            )
        )
        observer.emit(
            RunEvent(
                kind="agent_command_completed",
                message="codex planning finished with exit code 0",
                trial_id="demo-0001",
                phase="planning",
            )
        )

        self.assertEqual(len(observer.agent_lines), 2)
        self.assertIn("[agent]", observer.agent_lines[0])
        self.assertTrue(observer.agent_lines[0].endswith("mcp startup: no servers"))
        self.assertIn("[agent]", observer.agent_lines[1])
        self.assertTrue(observer.agent_lines[1].endswith("codex planning finished with exit code 0"))

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_routes_agent_heartbeat_to_agent_dialog(self) -> None:
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

        self.assertEqual(len(observer.agent_lines), 1)
        self.assertIn("[agent]", observer.agent_lines[0])
        self.assertTrue(observer.agent_lines[0].endswith("Agent process still running..."))

    @unittest.skipUnless(_RICH_AVAILABLE, "rich is not installed")
    def test_rich_run_observer_routes_bo_tool_calls_to_dedicated_panel(self) -> None:
        stream = io.StringIO()
        observer = RichRunObserver(stream=stream)

        observer.emit(
            RunEvent(
                kind="bo_tool_call",
                message="recent_trials: 4 trials",
                trial_id="demo-0001",
                phase="planning",
                source="bo.recent_trials",
            )
        )

        self.assertEqual(len(observer.bo_lines), 1)
        self.assertIn("[bo]", observer.bo_lines[0])
        self.assertTrue(observer.bo_lines[0].endswith("recent_trials: 4 trials"))


if __name__ == "__main__":
    unittest.main()
