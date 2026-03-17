from __future__ import annotations

from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "codex": {
        "preset": "codex",
        "runtime": "cli",
        "command": "codex",
        "args": [],
    },
    "claude_code": {
        "preset": "claude_code",
        "runtime": "cli",
        "command": "claude",
        "args": [],
    },
    "copilot": {
        "preset": "copilot",
        "runtime": "cli",
        "command": "copilot",
        "args": [],
    },
    "deepagents": {
        "preset": "deepagents",
        "runtime": "deepagents",
        "backend": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "api_key_env": "OPENAI_API_KEY",
        "model": "qwen2.5-coder:14b",
    },
    "custom": {
        "preset": "custom",
        "runtime": "cli",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_agent_profile(raw_agent: dict[str, Any]) -> dict[str, Any]:
    agent = dict(raw_agent or {})
    preset_name = str(agent.get("preset") or "codex").strip().lower()
    if preset_name not in PRESETS:
        if agent.get("command"):
            preset_name = "custom"
        else:
            known = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown agent preset '{preset_name}'. Expected one of: {known}")
    preset = PRESETS[preset_name]
    merged = _deep_merge(preset, agent)
    merged["preset"] = preset_name
    if preset_name == "custom":
        merged.setdefault("runtime", "cli")
    return merged
