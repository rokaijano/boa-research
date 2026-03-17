from __future__ import annotations

import re
from pathlib import Path

from .schema import CandidateMetadata, PatchDescriptor


NUMERIC_ASSIGNMENT_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.-]{1,63})\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
PY_SYMBOL_RE = re.compile(r"^\+\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
HUNK_SYMBOL_RE = re.compile(r"^@@ .* @@\s*(?P<context>.+?)\s*$")


def _extract_numeric_knobs(diff_text: str) -> dict[str, float]:
    knobs: dict[str, float] = {}
    for line in str(diff_text).splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for match in NUMERIC_ASSIGNMENT_RE.finditer(line):
            name = match.group("name").split(".")[-1]
            try:
                knobs.setdefault(name, float(match.group("value")))
            except ValueError:
                continue
    return knobs


def _extract_touched_symbols(diff_text: str) -> list[str]:
    symbols: list[str] = []
    for line in str(diff_text).splitlines():
        if line.startswith("@@"):
            match = HUNK_SYMBOL_RE.match(line)
            if match:
                context = match.group("context").strip()
                if context and context not in symbols:
                    symbols.append(context)
            continue
        match = PY_SYMBOL_RE.match(line)
        if match:
            symbol = match.group(1)
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def build_patch_descriptor(
    *,
    touched_files: list[str],
    diff_text: str,
    candidate: CandidateMetadata,
    parent_branch: str,
    parent_trial_id: str | None,
    budget_used: str,
    diff_path: Path,
) -> PatchDescriptor:
    numeric_knobs = dict(_extract_numeric_knobs(diff_text))
    numeric_knobs.update(candidate.numeric_knobs)
    touched_symbols = []
    for symbol in candidate.target_symbols + _extract_touched_symbols(diff_text):
        if symbol not in touched_symbols:
            touched_symbols.append(symbol)
    return PatchDescriptor(
        touched_files=touched_files,
        touched_symbols=touched_symbols,
        patch_category=candidate.patch_category,
        operation_type=candidate.operation_type,
        numeric_knobs=numeric_knobs,
        rationale_summary=candidate.rationale_summary,
        estimated_risk=float(candidate.estimated_risk),
        parent_branch=parent_branch,
        parent_trial_id=parent_trial_id,
        budget_used=budget_used,
        diff_path=str(diff_path),
    )
