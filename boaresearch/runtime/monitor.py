from __future__ import annotations

import sys
from collections import deque
from contextlib import AbstractContextManager
from datetime import datetime
from typing import TextIO

from .observer import NullRunObserver, RunEvent

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    Console = Group = Live = Panel = Table = None
    _RICH_AVAILABLE = False


def _short_ts(raw: str) -> str:
    try:
        return datetime.fromisoformat(raw).strftime("%H:%M:%S")
    except ValueError:
        return raw


def _scope_label(event: RunEvent) -> str:
    parts = []
    if event.trial_id:
        parts.append(event.trial_id)
    if event.phase:
        parts.append(event.phase)
    if event.stage_name:
        parts.append(event.stage_name)
    if event.source:
        parts.append(event.source)
    return " / ".join(parts)


def _event_tag(event: RunEvent) -> str:
    source = str(event.source or "")
    if event.kind.startswith("agent_"):
        return "[agent]"
    if source.startswith("agent."):
        return "[agent]"
    if source.startswith("stage.") or source.startswith("remote."):
        return "[term]"
    return "[boa]"


class PlainRunObserver(NullRunObserver, AbstractContextManager["PlainRunObserver"]):
    def __init__(self, *, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    def __enter__(self) -> PlainRunObserver:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        return None

    def emit(self, event: RunEvent) -> None:
        scope = _scope_label(event)
        prefix = f"[{_short_ts(event.timestamp)}]"
        if scope:
            prefix = f"{prefix} [{scope}]"
        self.stream.write(f"{prefix} {event.message}\n")
        self.stream.flush()


class RichRunObserver(NullRunObserver, AbstractContextManager["RichRunObserver"]):
    def __init__(self, *, stream: TextIO | None = None) -> None:
        if not _RICH_AVAILABLE:
            raise RuntimeError("rich is not available")
        self.stream = stream or sys.stderr
        self.console = Console(file=self.stream)
        self.events: deque[str] = deque(maxlen=14)
        self.agent_lines: deque[str] = deque(maxlen=14)
        self.terminal_lines: deque[str] = deque(maxlen=18)
        self.summary_lines: deque[str] = deque(maxlen=12)
        self.completed_trials: deque[tuple[str, str, str]] = deque(maxlen=8)
        self.current_trial_id: str | None = None
        self.current_phase: str | None = None
        self.current_stage: str | None = None
        self.current_status = "idle"
        self.live = Live(self._render(), console=self.console, refresh_per_second=8, transient=False)

    def __enter__(self) -> RichRunObserver:
        self.live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.live.stop()
        return None

    def emit(self, event: RunEvent) -> None:
        if event.trial_id:
            self.current_trial_id = event.trial_id
        if event.phase:
            self.current_phase = event.phase
        if event.stage_name:
            self.current_stage = event.stage_name
        if event.kind in {"run_started", "trial_started", "planning_started", "execution_started", "stage_started"}:
            self.current_status = event.message
        elif event.kind in {"trial_completed", "trial_failed", "run_completed", "run_stopped"}:
            self.current_status = event.message
        scope = _scope_label(event)
        scope_prefix = f"[{scope}] " if scope else ""
        rendered = f"{_short_ts(event.timestamp)} {_event_tag(event)} {scope_prefix}{event.message}"
        if event.kind == "process_output":
            self._record_process_output(event=event, rendered=rendered)
        elif event.kind in {
            "agent_prompt_sent",
            "agent_command_started",
            "agent_command_completed",
            "planning_bundle_ready",
            "execution_bundle_ready",
        }:
            self.agent_lines.append(rendered)
        elif event.kind in {
            "planning_completed",
            "planning_result",
            "execution_completed",
            "execution_result",
            "descriptor_ready",
            "stage_completed",
            "trial_completed",
            "trial_failed",
        }:
            self.summary_lines.append(rendered)
            self.events.append(rendered)
        else:
            self.events.append(rendered)
        if event.kind in {"trial_completed", "trial_failed"} and event.trial_id:
            status = str(event.metadata.get("acceptance_status", "unknown"))
            detail = str(event.metadata.get("detail", "") or "")
            self.completed_trials.append((event.trial_id, status, detail))
        self.live.update(self._render(), refresh=True)

    def _record_process_output(self, *, event: RunEvent, rendered: str) -> None:
        source = str(event.source or "")
        if source.startswith("stage.") or source.startswith("remote."):
            self.terminal_lines.append(rendered)
            return
        if source.startswith("agent."):
            return
        self.events.append(rendered)

    def _render(self):
        header = Table.grid(expand=True)
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_row(
            f"[bold]Trial[/bold]: {self.current_trial_id or '-'}",
            f"[bold]Phase[/bold]: {self.current_phase or '-'}",
        )
        header.add_row(
            f"[bold]Stage[/bold]: {self.current_stage or '-'}",
            f"[bold]Status[/bold]: {self.current_status}",
        )

        trial_table = Table(expand=True, box=None, show_header=True, header_style="bold")
        trial_table.add_column("Trial", ratio=2)
        trial_table.add_column("Status", ratio=2)
        trial_table.add_column("Detail", ratio=5)
        if self.completed_trials:
            for trial_id, status, detail in self.completed_trials:
                trial_table.add_row(trial_id, status, detail or "-")
        else:
            trial_table.add_row("-", "-", "No completed trials yet")

        event_table = Table(expand=True, box=None, show_header=False)
        event_table.add_column(ratio=1)
        if self.events:
            for line in self.events:
                event_table.add_row(line)
        else:
            event_table.add_row("Waiting for BOA events...")

        summary_table = Table(expand=True, box=None, show_header=False)
        summary_table.add_column(ratio=1)
        if self.summary_lines:
            for line in self.summary_lines:
                summary_table.add_row(line)
        else:
            summary_table.add_row("No structured results yet")

        agent_table = Table(expand=True, box=None, show_header=False)
        agent_table.add_column(ratio=1)
        if self.agent_lines:
            for line in self.agent_lines:
                agent_table.add_row(line)
        else:
            agent_table.add_row("No agent dialog yet")

        terminal_table = Table(expand=True, box=None, show_header=False)
        terminal_table.add_column(ratio=1)
        if self.terminal_lines:
            for line in self.terminal_lines:
                terminal_table.add_row(line)
        else:
            terminal_table.add_row("No stage terminal output yet")

        return Group(
            Panel(header, title="BOA Run"),
            Panel(trial_table, title="Recent Trials"),
            Panel(summary_table, title="Prompt / Result Summary"),
            Panel(agent_table, title="Agent Dialog"),
            Panel(terminal_table, title="Inner Terminal"),
            Panel(event_table, title="Detailed Log"),
        )


def build_run_observer(*, stream: TextIO | None = None, interactive: bool | None = None):
    target = stream or sys.stderr
    use_interactive = target.isatty() if interactive is None else interactive
    if use_interactive and _RICH_AVAILABLE:
        return RichRunObserver(stream=target)
    return PlainRunObserver(stream=target)
