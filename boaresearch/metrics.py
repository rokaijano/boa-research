from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .schema import MetricConfig


class MetricExtractionError(RuntimeError):
    pass


def _resolve_json_key(payload, dotted_key: str):
    current = payload
    for part in str(dotted_key).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise MetricExtractionError(f"Unable to resolve json_key '{dotted_key}'")
    return current


def _first_float(text: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        raise MetricExtractionError("No numeric value found")
    return float(match.group(0))


def _metric_file_value(text: str, metric_name: str) -> float:
    stripped = str(text).strip()
    if not stripped:
        raise MetricExtractionError("Metric file is empty")
    for line in stripped.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == metric_name:
                return float(value.strip())
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == metric_name:
                return float(value.strip())
    return _first_float(stripped)


def collect_logs(artifact_dir: Path) -> dict[str, str]:
    stdout_blocks: list[str] = []
    stderr_blocks: list[str] = []
    for stdout_path in sorted(artifact_dir.glob("command_*.stdout")):
        stdout_blocks.append(stdout_path.read_text(encoding="utf-8"))
    for stderr_path in sorted(artifact_dir.glob("command_*.stderr")):
        stderr_blocks.append(stderr_path.read_text(encoding="utf-8"))
    return {
        "stdout": "\n".join(stdout_blocks),
        "stderr": "\n".join(stderr_blocks),
        "combined": "\n".join(stdout_blocks + stderr_blocks),
    }


def extract_metrics(*, artifact_dir: Path, metrics: Iterable[MetricConfig]) -> dict[str, float]:
    logs = collect_logs(artifact_dir)
    extracted: dict[str, float] = {}
    for metric in metrics:
        source = metric.source
        value: float | None = None
        if source == "regex":
            target = str(metric.target or "combined").strip().lower()
            haystack = logs.get(target, logs["combined"])
            match = re.search(str(metric.pattern or ""), haystack, flags=re.MULTILINE)
            if match:
                try:
                    group_value = match.group(int(metric.group))
                except (IndexError, ValueError):
                    group_value = match.group(str(metric.group))
                value = float(group_value)
        else:
            if not metric.path:
                raise MetricExtractionError(f"Metric '{metric.name}' is missing a path")
            captured_path = artifact_dir / "captured" / metric.path
            if captured_path.exists():
                text = captured_path.read_text(encoding="utf-8")
                if source == "json_file":
                    payload = json.loads(text)
                    value = float(_resolve_json_key(payload, str(metric.json_key)))
                elif source == "metric_file":
                    value = _metric_file_value(text, metric.name)
        if value is None:
            if metric.required:
                raise MetricExtractionError(f"Failed to extract required metric '{metric.name}'")
            continue
        extracted[metric.name] = value
    return extracted
