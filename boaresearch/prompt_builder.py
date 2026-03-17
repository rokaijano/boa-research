from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import AgentContext, OPERATION_TYPES, PATCH_CATEGORIES


CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "hypothesis",
        "rationale_summary",
        "patch_category",
        "operation_type",
        "estimated_risk",
    ],
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale_summary": {"type": "string"},
        "patch_category": {"type": "string", "enum": sorted(PATCH_CATEGORIES)},
        "operation_type": {"type": "string", "enum": sorted(OPERATION_TYPES)},
        "estimated_risk": {"type": "number"},
        "target_symbols": {"type": "array", "items": {"type": "string"}},
        "numeric_knobs": {"type": "object", "additionalProperties": {"type": "number"}},
        "notes": {"type": ["string", "null"]},
    },
}


CANDIDATE_EXAMPLE = {
    "hypothesis": "A shorter warmup with stronger weight decay should reduce overfitting.",
    "rationale_summary": "Adjust optimizer scheduling and tighten regularization around the training loop.",
    "patch_category": "optimizer",
    "operation_type": "replace",
    "estimated_risk": 0.28,
    "target_symbols": ["Trainer.train_epoch"],
    "numeric_knobs": {"learning_rate": 0.0002, "weight_decay": 0.02, "warmup_steps": 250},
    "notes": "Keep dataset code unchanged.",
}


def _read_text(path: Path, *, limit: int | None = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"<failed to read {path}: {type(exc).__name__}: {exc}>"
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated] ...\n"


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_agent_system_prompt(*, repo_root: Path, context: AgentContext) -> str:
    sections = [
        "You are the patch-authoring agent inside BOA (Bayesian Optimized Agents).",
        "BOA decides search direction, branch lineage, evaluation stages, and acceptance. You only author a candidate patch in the provided trial worktree.",
        "Respect the repository contract in boa.md and the machine-enforced path guardrails.",
        f"Accepted branch: {context.accepted_branch}",
        f"Trial branch: {context.trial_branch}",
        f"Allowed paths: {', '.join(context.allowed_paths) if context.allowed_paths else '<all tracked paths>'}",
        f"Protected paths: {', '.join(context.protected_paths) if context.protected_paths else '<none>'}",
        f"Primary contract ({_display_path(context.boa_md_path, repo_root)}):\n```\n{_read_text(context.boa_md_path, limit=None)}\n```",
    ]
    agents_md = repo_root / "AGENTS.md"
    if agents_md.exists():
        sections.append(f"Supplemental repo instructions (AGENTS.md):\n```\n{_read_text(agents_md)}\n```")
    if context.extra_context_files:
        for path in context.extra_context_files:
            sections.append(f"Extra context ({_display_path(path, repo_root)}):\n```\n{_read_text(path)}\n```")
    return "\n\n".join(sections)


def build_agent_task(context: AgentContext) -> str:
    recent_lines: list[str] = []
    for trial in context.recent_trials:
        recent_lines.append(
            " | ".join(
                [
                    trial.trial_id,
                    trial.acceptance_status,
                    trial.canonical_stage or "-",
                    "" if trial.canonical_score is None else f"{trial.canonical_score:.6f}",
                    trial.descriptor.patch_category if trial.descriptor else "-",
                    trial.candidate.rationale_summary if trial.candidate else "-",
                ]
            )
        )
    if not recent_lines:
        recent_lines = ["<no prior trials recorded>"]

    hints = context.search_decision.prompt_hints or ["<no additional search hints>"]
    sections = [
        f"Trial id: {context.trial_id}",
        context.objective_summary,
        f"Search policy: {context.search_decision.policy}",
        f"Parent branch: {context.search_decision.parent_branch}",
        f"Parent trial: {context.search_decision.parent_trial_id or 'none'}",
        f"Patch category hint: {context.search_decision.patch_category_hint or 'none'}",
        "Search hints:",
        "\n".join(f"- {hint}" for hint in hints),
        "Recent trials:",
        "\n".join(recent_lines),
        "Preflight commands:",
        "\n".join(context.preflight_commands) if context.preflight_commands else "<none>",
        "Requirements:",
        "1. Edit only inside the provided trial worktree.",
        "2. Do not touch protected paths.",
        "3. Keep the candidate coherent with the search guidance.",
        "4. Run the configured preflight commands before finalizing.",
        "5. Write exactly one candidate metadata JSON object to the provided output path and print the same JSON to stdout as a fallback.",
        "6. Do not commit or push; BOA owns branching, commits, and acceptance.",
    ]
    return "\n\n".join(sections)


def build_cli_prompt_bundle(*, repo_root: Path, context: AgentContext) -> dict[str, str]:
    system_prompt = build_agent_system_prompt(repo_root=repo_root, context=context)
    task_prompt = build_agent_task(context)
    output_instructions = "\n\n".join(
        [
            "Candidate metadata schema:",
            f"```json\n{json.dumps(CANDIDATE_SCHEMA, indent=2, sort_keys=True)}\n```",
            "Example candidate metadata:",
            f"```json\n{json.dumps(CANDIDATE_EXAMPLE, indent=2, sort_keys=True)}\n```",
        ]
    )
    combined = "\n\n".join(
        [
            "SYSTEM PROMPT",
            system_prompt,
            "TASK PROMPT",
            task_prompt,
            output_instructions,
        ]
    )
    return {
        "system_prompt": system_prompt,
        "task_prompt": task_prompt,
        "combined_prompt": combined,
        "candidate_schema": json.dumps(CANDIDATE_SCHEMA, indent=2, sort_keys=True),
        "candidate_example": json.dumps(CANDIDATE_EXAMPLE, indent=2, sort_keys=True),
    }
