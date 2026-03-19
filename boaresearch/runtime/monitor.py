from __future__ import annotations

import re
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
    if event.kind.startswith("bo_") or source.startswith("bo."):
        return "[bo]"
    if source.startswith("agent.") and event.phase == "execution":
        return "[term]"
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
        self.activity_lines: deque[str] = deque(maxlen=16)
        self.error_lines: deque[str] = deque(maxlen=8)
        self.trial_sequence: list[str] = []
        self.trials: dict[str, dict[str, object]] = {}
        self.current_trial_id: str | None = None
        self.current_phase: str | None = None
        self.current_stage: str | None = None
        self.current_status = "idle"
        self.transient_status: str | None = None
        self.transient_pulse = 0
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
            self._ensure_trial(event.trial_id)
        if event.phase:
            self.current_phase = event.phase
        if event.stage_name:
            self.current_stage = event.stage_name
        if event.kind in {"run_started", "trial_started", "planning_started", "execution_started", "stage_started"}:
            self.current_status = event.message
            self.transient_status = None
        elif event.kind in {"trial_completed", "trial_failed", "run_completed", "run_stopped"}:
            self.current_status = event.message
            self.transient_status = None
        self._update_trial_state(event)
        self._record_event(event)
        self.live.update(self._render(), refresh=True)

    def _ensure_trial(self, trial_id: str) -> dict[str, object]:
        if trial_id not in self.trials:
            self.trials[trial_id] = {
                "trial_id": trial_id,
                "branch_name": "",
                "parent_branch": "",
                "parent_trial_id": None,
                "acceptance_status": "running",
                "canonical_stage": None,
                "canonical_score": None,
                "primary_metric": None,
                "detail": "",
            }
            self.trial_sequence.append(trial_id)
        return self.trials[trial_id]

    def _update_trial_state(self, event: RunEvent) -> None:
        if not event.trial_id:
            return
        record = self._ensure_trial(event.trial_id)
        if event.stage_name:
            record["canonical_stage"] = event.stage_name
        for key in (
            "branch_name",
            "parent_branch",
            "parent_trial_id",
            "acceptance_status",
            "canonical_stage",
            "canonical_score",
            "primary_metric",
        ):
            if key in event.metadata and event.metadata.get(key) is not None:
                record[key] = event.metadata.get(key)
        if event.kind == "stage_completed":
            if event.metadata.get("score") is not None:
                record["canonical_score"] = event.metadata.get("score")
            if event.metadata.get("primary_metric") is not None:
                record["primary_metric"] = event.metadata.get("primary_metric")
            if event.metadata.get("branch_name"):
                record["branch_name"] = event.metadata.get("branch_name")
        if event.kind == "trial_failed":
            record["detail"] = str(event.metadata.get("detail") or "")
        elif event.kind == "trial_completed":
            record["detail"] = ""

    def _rendered_line(self, event: RunEvent) -> str:
        scope = _scope_label(event)
        scope_prefix = f"[{scope}] " if scope else ""
        return f"{_short_ts(event.timestamp)} {_event_tag(event)} {scope_prefix}{event.message}"

    @staticmethod
    def _is_usage_noise(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if stripped.startswith(("│", "└", "{", "}", '"', "[", "]", ",")):
            return True
        if stripped.startswith(("Total usage est:", "API time spent:", "Total session time:", "Total code changes:")):
            return True
        if stripped.startswith("Breakdown by AI model:"):
            return True
        if re.match(r"^[A-Za-z0-9_.-]+\s+\d", stripped):
            return True
        return False

    @staticmethod
    def _normalize_agent_output(message: str) -> str | None:
        stripped = message.strip()
        if RichRunObserver._is_usage_noise(stripped):
            return None
        if stripped.startswith(("● ", "✗ ", "Reading ", "Applying ")):
            return stripped
        if any(token in stripped.lower() for token in ("retry", "error", "failed", "invalid")):
            return stripped
        return None

    @staticmethod
    def _normalize_terminal_output(message: str) -> str | None:
        stripped = message.strip()
        if not stripped:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?%", stripped):
            return None
        return stripped

    def _append_activity(self, line: str) -> None:
        self.activity_lines.append(line)

    def _append_error(self, line: str) -> None:
        self.error_lines.append(line)

    def _record_process_output(self, *, event: RunEvent) -> None:
        source = str(event.source or "")
        rendered = self._rendered_line(event)
        if source.startswith("stage.") or source.startswith("remote."):
            normalized = self._normalize_terminal_output(event.message)
            if normalized:
                self._append_activity(f"{_short_ts(event.timestamp)} [term] {normalized}")
            if source.endswith(".stderr") and normalized:
                self._append_error(f"{_short_ts(event.timestamp)} [term] {normalized}")
            return
        if source.startswith("agent."):
            normalized = self._normalize_agent_output(event.message)
            if normalized:
                target = self._append_error if any(token in normalized.lower() for token in ("error", "failed", "invalid")) else self._append_activity
                target(f"{_short_ts(event.timestamp)} {_event_tag(event)} {normalized}")
            return
        self._append_activity(rendered)

    def _record_event(self, event: RunEvent) -> None:
        rendered = self._rendered_line(event)
        if event.kind == "process_output":
            self._record_process_output(event=event)
            return
        if event.kind == "bo_tool_call":
            self._append_activity(rendered)
            return
        if event.kind == "process_waiting":
            self.transient_status = event.message
            self.transient_pulse = (self.transient_pulse + 1) % 8
            return
        if event.kind in {
            "planning_bundle_ready",
            "execution_bundle_ready",
            "agent_prompt_sent",
            "agent_command_started",
            "agent_command_completed",
            "planning_result",
            "execution_result",
            "descriptor_ready",
            "stage_started",
            "stage_completed",
            "trial_started",
            "trial_completed",
            "baseline_started",
            "baseline_completed",
        }:
            self._append_activity(rendered)
        if event.kind in {"trial_failed", "stage_failed", "run_stopped"}:
            self._append_error(rendered)
        detail = str(event.metadata.get("detail") or "").strip()
        if detail:
            self._append_error(f"{_short_ts(event.timestamp)} [detail] {detail}")

    def _lineage_depth(self, trial_id: str) -> int:
        depth = 0
        current = self.trials.get(trial_id, {})
        seen: set[str] = set()
        while True:
            parent_trial_id = str(current.get("parent_trial_id") or "")
            if not parent_trial_id:
                status = str(current.get("acceptance_status") or "")
                return depth if status == "baseline" else depth + 1
            if parent_trial_id in seen:
                return depth + 1
            seen.add(parent_trial_id)
            current = self.trials.get(parent_trial_id, {})
            depth += 1

    def _best_trial(self) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for trial_id in self.trial_sequence:
            record = self.trials.get(trial_id)
            if record is None:
                continue
            metric = record.get("primary_metric")
            if metric is None:
                continue
            metric_value = float(metric)
            if best is None or metric_value > best[1]:
                best = (trial_id, metric_value)
        return best

    def _render_header(self):
        header = Table.grid(expand=True)
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        current = self.trials.get(self.current_trial_id or "", {})
        best = self._best_trial()
        current_metric = current.get("primary_metric")
        current_parent = str(current.get("parent_trial_id") or current.get("parent_branch") or "-")
        header.add_row(
            f"[bold]Trial[/bold]: {self.current_trial_id or '-'}",
            f"[bold]Phase[/bold]: {self.current_phase or '-'}",
        )
        header.add_row(
            f"[bold]Stage[/bold]: {self.current_stage or '-'}",
            f"[bold]Status[/bold]: {self.current_status}",
        )
        header.add_row(
            f"[bold]Parent[/bold]: {current_parent or '-'}",
            f"[bold]Accuracy[/bold]: {'-' if current_metric is None else f'{float(current_metric):.4%}'}",
        )
        header.add_row(
            "[bold]Best[/bold]: " + ("-" if best is None else f"{best[0]} ({best[1]:.4%})"),
            f"[bold]Branch[/bold]: {current.get('branch_name') or '-'}",
        )
        if self.transient_status:
            filled = self.transient_pulse + 1
            bar = "[" + ("=" * filled) + (" " * (8 - filled)) + "]"
            header.add_row(
                f"[bold]Activity[/bold]: {self.transient_status}",
                f"[bold]Pulse[/bold]: {bar}",
            )
        return header

    def _render_lineage(self):
        lineage = Table(expand=True, box=None, show_header=True, header_style="bold")
        lineage.add_column("Lineage", ratio=4)
        lineage.add_column("Accuracy", ratio=2, justify="right")
        lineage.add_column("Status", ratio=2)
        for trial_id in self.trial_sequence:
            record = self.trials.get(trial_id)
            if record is None:
                continue
            depth = self._lineage_depth(trial_id)
            prefix = "  " * max(0, depth - 1)
            marker = "*" if trial_id == self.current_trial_id else " "
            label = "accepted baseline" if str(record.get("acceptance_status")) == "baseline" else trial_id
            lineage_label = f"{marker} {prefix}{'└─ ' if depth else ''}{label}"
            metric = record.get("primary_metric")
            lineage.add_row(
                lineage_label,
                "-" if metric is None else f"{float(metric):.4%}",
                str(record.get("acceptance_status") or "-"),
            )
        if not self.trial_sequence:
            lineage.add_row("No trials yet", "-", "-")
        return lineage

    def _render(self):
        activity_table = Table(expand=True, box=None, show_header=False)
        activity_table.add_column(ratio=1)
        if self.activity_lines:
            for line in self.activity_lines:
                activity_table.add_row(line)
        else:
            activity_table.add_row("Waiting for activity...")

        error_table = Table(expand=True, box=None, show_header=False)
        error_table.add_column(ratio=1)
        if self.error_lines:
            for line in self.error_lines:
                error_table.add_row(line)
        else:
            error_table.add_row("No errors or warnings yet")

        return Group(
            Panel(self._render_header(), title="BOA Run"),
            Panel(self._render_lineage(), title="Lineage And Accuracy"),
            Panel(activity_table, title="Activity"),
            Panel(error_table, title="Errors"),
        )


def build_run_observer(*, stream: TextIO | None = None, interactive: bool | None = None):
    target = stream or sys.stderr
    use_interactive = target.isatty() if interactive is None else interactive
    if use_interactive and _RICH_AVAILABLE:
        return RichRunObserver(stream=target)
    return PlainRunObserver(stream=target)
