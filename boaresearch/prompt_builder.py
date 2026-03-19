from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from .prompt_templates import render_prompt_template
from .schema import AgentExecutionContext, AgentPlanningContext, OPERATION_TYPES, PATCH_CATEGORIES


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypothesis",
        "rationale_summary",
        "selected_parent_branch",
        "selected_parent_trial_id",
        "patch_category",
        "operation_type",
        "estimated_risk",
        "target_symbols",
        "numeric_knobs",
        "notes",
        "informed_by_call_ids",
    ],
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale_summary": {"type": "string"},
        "selected_parent_branch": {"type": "string"},
        "selected_parent_trial_id": {"type": ["string", "null"]},
        "patch_category": {"type": "string", "enum": sorted(PATCH_CATEGORIES)},
        "operation_type": {"type": "string", "enum": sorted(OPERATION_TYPES)},
        "estimated_risk": {"type": "number"},
        "target_symbols": {"type": "array", "items": {"type": "string"}},
        "numeric_knobs": {"type": "object", "additionalProperties": {"type": "number"}},
        "notes": {"type": ["string", "null"]},
        "informed_by_call_ids": {"type": "array", "items": {"type": "string"}},
    },
}


PLAN_EXAMPLE = {
    "hypothesis": "A shorter warmup and stronger weight decay should reduce overfitting.",
    "rationale_summary": "Start from the best optimizer lineage and probe a tighter regularization schedule.",
    "selected_parent_branch": "boa/demo/trial/demo-0007",
    "selected_parent_trial_id": "demo-0007",
    "patch_category": "optimizer",
    "operation_type": "replace",
    "estimated_risk": 0.28,
    "target_symbols": ["Trainer.train_epoch"],
    "numeric_knobs": {"learning_rate": 0.0002, "weight_decay": 0.02, "warmup_steps": 250},
    "notes": "Keep dataset code unchanged.",
    "informed_by_call_ids": ["boa-call-abc123def456"],
}


CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypothesis",
        "rationale_summary",
        "patch_category",
        "operation_type",
        "estimated_risk",
        "target_symbols",
        "numeric_knobs",
        "notes",
        "informed_by_call_ids",
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
        "informed_by_call_ids": {"type": "array", "items": {"type": "string"}},
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
    "informed_by_call_ids": ["boa-call-abc123def456", "boa-call-fed654cba321"],
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


def _supplemental_sections(*, repo_root: Path, extra_context_files: list[Path]) -> str:
    sections: list[str] = []
    agents_md = repo_root / "AGENTS.md"
    if agents_md.exists():
        sections.append(f"Supplemental repo instructions (AGENTS.md):\n```\n{_read_text(agents_md)}\n```")
    for path in extra_context_files:
        sections.append(f"Extra context ({_display_path(path, repo_root)}):\n```\n{_read_text(path)}\n```")
    return "\n\n".join(sections) if sections else "<no supplemental repo instructions>"


def _bootstrap_tool_sections(calls: list[Any]) -> str:
    if not calls:
        return "<no bootstrap BOA tool results provided>"
    sections: list[str] = []
    for call in calls:
        sections.append(
            "\n".join(
                [
                    f"Call id: {call.call_id}",
                    f"Tool: {call.tool_name}",
                    f"Request: {json.dumps(call.request, sort_keys=True)}",
                    f"Response: {json.dumps(call.response, indent=2, sort_keys=True)}",
                ]
            )
        )
    return "\n\n".join(sections)


def _tool_command_display() -> str:
    return f'"{sys.executable}" -m boaresearch.cli'


def _tool_shell_examples(tool_command: str) -> dict[str, str]:
    if os.name == "nt":
        return {
            "tool_recent_trials_example": f"echo '{{}}' | {tool_command} tools recent-trials",
            "tool_list_lineage_options_example": f"echo '{{}}' | {tool_command} tools list-lineage-options",
            "tool_suggest_parents_example": (
                f"echo '{{\"patch_category\":\"optimizer\",\"operation_type\":\"replace\",\"estimated_risk\":0.25}}' | "
                f"{tool_command} tools suggest-parents"
            ),
        }
    return {
        "tool_recent_trials_example": f"printf '{{}}' | {tool_command} tools recent-trials",
        "tool_list_lineage_options_example": f"printf '{{}}' | {tool_command} tools list-lineage-options",
        "tool_suggest_parents_example": (
            f"printf '{{\"patch_category\":\"optimizer\",\"operation_type\":\"replace\",\"estimated_risk\":0.25}}' | "
            f"{tool_command} tools suggest-parents"
        ),
    }


def build_planning_system_prompt(
    *,
    repo_root: Path,
    context: AgentPlanningContext,
    tool_command: str = "boa",
) -> str:
    del tool_command
    prompt_tool_command = _tool_command_display()
    tool_examples = _tool_shell_examples(prompt_tool_command)
    return render_prompt_template(
        "agent",
        "planning_system.md",
        accepted_branch=context.accepted_branch,
        allowed_paths=", ".join(context.allowed_paths) if context.allowed_paths else "<all tracked paths>",
        protected_paths=", ".join(context.protected_paths) if context.protected_paths else "<none>",
        boa_md_display_path=_display_path(context.boa_md_path, repo_root),
        boa_md_text=_read_text(context.boa_md_path, limit=None),
        supplemental_sections=_supplemental_sections(repo_root=repo_root, extra_context_files=context.extra_context_files),
        tool_command=prompt_tool_command,
        tool_command_quoted=prompt_tool_command,
        **tool_examples,
    )


def build_execution_system_prompt(
    *,
    repo_root: Path,
    context: AgentExecutionContext,
    tool_command: str = "boa",
) -> str:
    del tool_command
    prompt_tool_command = _tool_command_display()
    tool_examples = _tool_shell_examples(prompt_tool_command)
    return render_prompt_template(
        "agent",
        "execution_system.md",
        accepted_branch=context.accepted_branch,
        allowed_paths=", ".join(context.allowed_paths) if context.allowed_paths else "<all tracked paths>",
        protected_paths=", ".join(context.protected_paths) if context.protected_paths else "<none>",
        boa_md_display_path=_display_path(context.boa_md_path, repo_root),
        boa_md_text=_read_text(context.boa_md_path, limit=None),
        supplemental_sections=_supplemental_sections(repo_root=repo_root, extra_context_files=context.extra_context_files),
        parent_branch=context.parent_branch,
        trial_branch=context.trial_branch,
        tool_command=prompt_tool_command,
        tool_command_quoted=prompt_tool_command,
        **tool_examples,
    )


def build_planning_task(context: AgentPlanningContext) -> str:
    recent_lines = []
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
    return render_prompt_template(
        "agent",
        "planning_task.md",
        trial_id=context.trial_id,
        objective_summary=context.objective_summary,
        plan_output_path=str(context.plan_output_path),
        recent_trials="\n".join(recent_lines) if recent_lines else "<no prior trials recorded>",
        bootstrap_tool_calls=_bootstrap_tool_sections(context.bootstrap_tool_calls),
    )


def build_execution_task(context: AgentExecutionContext) -> str:
    recent_lines = []
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
    return render_prompt_template(
        "agent",
        "execution_task.md",
        trial_id=context.trial_id,
        objective_summary=context.objective_summary,
        parent_branch=context.parent_branch,
        parent_trial_id=context.parent_trial_id or "none",
        plan_output_path=str(context.plan_output_path),
        candidate_output_path=str(context.candidate_output_path),
        candidate_plan_json=json.dumps(context.candidate_plan.__dict__, indent=2, sort_keys=True),
        recent_trials="\n".join(recent_lines) if recent_lines else "<no prior trials recorded>",
        bootstrap_tool_calls=_bootstrap_tool_sections(context.bootstrap_tool_calls),
        preflight_commands="\n".join(context.preflight_commands) if context.preflight_commands else "<none>",
    )


def build_cli_planning_bundle(
    *,
    repo_root: Path,
    context: AgentPlanningContext,
    tool_command: str = "boa",
) -> dict[str, str]:
    system_prompt = build_planning_system_prompt(repo_root=repo_root, context=context, tool_command=tool_command)
    task_prompt = build_planning_task(context)
    output_instructions = "\n\n".join(
        [
            "Candidate plan schema:",
            f"```json\n{json.dumps(PLAN_SCHEMA, indent=2, sort_keys=True)}\n```",
            "Example candidate plan:",
            f"```json\n{json.dumps(PLAN_EXAMPLE, indent=2, sort_keys=True)}\n```",
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
        "plan_schema": json.dumps(PLAN_SCHEMA, indent=2, sort_keys=True),
        "plan_example": json.dumps(PLAN_EXAMPLE, indent=2, sort_keys=True),
    }


def build_cli_execution_bundle(
    *,
    repo_root: Path,
    context: AgentExecutionContext,
    tool_command: str = "boa",
) -> dict[str, str]:
    system_prompt = build_execution_system_prompt(repo_root=repo_root, context=context, tool_command=tool_command)
    task_prompt = build_execution_task(context)
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
