from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from .agent_presets import resolve_agent_profile
from .schema import (
    CONFIG_SCHEMA_VERSION,
    AgentConfig,
    BoaConfig,
    GitAuthConfig,
    GuardrailConfig,
    LocalRunnerConfig,
    MetricConfig,
    ObjectiveConfig,
    RunConfig,
    RunnerConfig,
    RunnerStageConfig,
    SSHRunnerConfig,
    SearchConfig,
)


ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "run",
    "agent",
    "guardrails",
    "git_auth",
    "runner",
    "metrics",
    "objective",
    "search",
}
STAGE_NAMES = ("scout", "confirm", "promoted")


def find_repo_root(start: Path) -> Path:
    probe = start.resolve()
    for candidate in (probe, *probe.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate git repo root from {start}")


def _to_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _resolve_local_path(path: Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (repo_root / path).resolve()


def _normalize_path_prefixes(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in values or []:
        cleaned = str(raw).replace("\\", "/").strip().lstrip("./").rstrip("/")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _normalize_str_list(values: list[Any] | None) -> list[str]:
    return [str(value) for value in list(values or [])]


def _normalize_str_map(values: dict[str, Any] | None, *, field_name: str) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return {str(key): str(value) for key, value in values.items()}


def _build_stage_config(raw: dict[str, Any] | None, *, default_enabled: bool, default_timeout: int) -> RunnerStageConfig:
    data = dict(raw or {})
    return RunnerStageConfig(
        enabled=bool(data.get("enabled", default_enabled)),
        commands=_normalize_str_list(data.get("commands")),
        timeout_seconds=int(data.get("timeout_seconds", default_timeout)),
        env=_normalize_str_map(data.get("env"), field_name="runner stage env"),
    )


def _build_runner_config(raw: dict[str, Any] | None) -> RunnerConfig:
    data = dict(raw or {})
    return RunnerConfig(
        mode=str(data.get("mode", "local")),
        local=LocalRunnerConfig(**dict(data.get("local", {}) or {})),
        ssh=SSHRunnerConfig(**dict(data.get("ssh", {}) or {})),
        scout=_build_stage_config(data.get("scout"), default_enabled=True, default_timeout=1800),
        confirm=_build_stage_config(data.get("confirm"), default_enabled=False, default_timeout=3600),
        promoted=_build_stage_config(data.get("promoted"), default_enabled=False, default_timeout=7200),
    )


def _build_metric_configs(raw: Any) -> list[MetricConfig]:
    metrics = list(raw or [])
    out: list[MetricConfig] = []
    for item in metrics:
        if not isinstance(item, dict):
            raise ValueError("Each [[metrics]] entry must be a table")
        out.append(MetricConfig(**item))
    return out


def _load_toml_dict(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"TOML root must be a table: {path}")
    return data


def _build_config(raw: dict[str, Any], repo_root: Path, config_path: Path, boa_md_path: Path) -> BoaConfig:
    unexpected = sorted(key for key in raw.keys() if key not in ALLOWED_TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(f"Unsupported boa.config section(s): {', '.join(unexpected)}")
    schema_version = int(raw.get("schema_version", 0))
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"boa.config is not a current BOA config: expected schema_version = {CONFIG_SCHEMA_VERSION}, found {schema_version}"
        )
    agent_raw = resolve_agent_profile(dict(raw.get("agent", {}) or {}))
    cfg = BoaConfig(
        schema_version=schema_version,
        run=RunConfig(**dict(raw.get("run", {}) or {})),
        agent=AgentConfig(**agent_raw),
        guardrails=GuardrailConfig(**dict(raw.get("guardrails", {}) or {})),
        git_auth=GitAuthConfig(**dict(raw.get("git_auth", {}) or {})),
        runner=_build_runner_config(dict(raw.get("runner", {}) or {})),
        metrics=_build_metric_configs(raw.get("metrics")),
        objective=ObjectiveConfig(**dict(raw.get("objective", {}) or {})),
        search=SearchConfig(**dict(raw.get("search", {}) or {})),
        repo_root=repo_root,
        config_path=config_path,
        boa_md_path=boa_md_path,
    )
    cfg.run.worktree_path = _resolve_local_path(_to_path(cfg.run.worktree_path), repo_root)
    cfg.run.stop_file = _resolve_local_path(_to_path(cfg.run.stop_file), repo_root)
    cfg.runner.ssh.key_path = _resolve_local_path(_to_path(cfg.runner.ssh.key_path), repo_root)
    cfg.agent.extra_context_files = [
        _resolve_local_path(_to_path(path), repo_root)  # type: ignore[arg-type]
        for path in list(cfg.agent.extra_context_files)
    ]
    cfg.agent.args = _normalize_str_list(cfg.agent.args)
    cfg.agent.env = _normalize_str_map(cfg.agent.env, field_name="agent.env")
    cfg.agent.provider_options = dict(cfg.agent.provider_options or {})
    cfg.guardrails.allowed_paths = _normalize_path_prefixes(cfg.guardrails.allowed_paths)
    cfg.guardrails.protected_paths = _normalize_path_prefixes(cfg.guardrails.protected_paths)
    cfg.guardrails.preflight_commands = _normalize_str_list(cfg.guardrails.preflight_commands)
    cfg.runner.local.env = _normalize_str_map(cfg.runner.local.env, field_name="runner.local.env")
    cfg.runner.ssh.env = _normalize_str_map(cfg.runner.ssh.env, field_name="runner.ssh.env")
    return cfg


def enabled_stages(cfg: BoaConfig) -> list[str]:
    stages = ["scout"]
    if cfg.runner.confirm.enabled:
        stages.append("confirm")
    if cfg.runner.promoted.enabled:
        stages.append("promoted")
    return stages


def validate_config(cfg: BoaConfig) -> BoaConfig:
    if cfg.schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")

    tag = str(cfg.run.tag).strip()
    if not tag:
        raise ValueError("run.tag is required")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        raise ValueError("run.tag must contain only letters, numbers, dot, underscore, or hyphen")
    cfg.run.tag = tag
    if cfg.run.max_trials < 1:
        raise ValueError("run.max_trials must be >= 1")
    if cfg.run.max_consecutive_failures < 1:
        raise ValueError("run.max_consecutive_failures must be >= 1")
    if cfg.run.accepted_branch is None:
        cfg.run.accepted_branch = f"boa/{tag}/accepted"
    if cfg.run.worktree_path is None:
        cfg.run.worktree_path = (cfg.repo_root / ".boa" / "worktrees" / tag).resolve()
    if cfg.run.stop_file is None:
        cfg.run.stop_file = (cfg.repo_root / ".boa" / "STOP").resolve()

    runtime = str(cfg.agent.runtime).strip().lower()
    if runtime not in {"cli", "deepagents"}:
        raise ValueError("agent.runtime must be one of: cli, deepagents")
    cfg.agent.runtime = runtime
    if runtime == "cli" and not str(cfg.agent.command or "").strip():
        raise ValueError("agent.command is required for CLI runtimes")
    if cfg.agent.prepare_timeout_seconds < 1:
        raise ValueError("agent.prepare_timeout_seconds must be >= 1")
    if cfg.agent.max_agent_steps < 1:
        raise ValueError("agent.max_agent_steps must be >= 1")
    if cfg.agent.max_output_tokens < 1:
        raise ValueError("agent.max_output_tokens must be >= 1")
    for path in cfg.agent.extra_context_files:
        if path is None or not path.exists():
            raise FileNotFoundError(f"agent.extra_context_files entry not found: {path}")

    if cfg.git_auth.persist_credentials:
        raise ValueError("git_auth.persist_credentials=true is not supported")
    if not str(cfg.git_auth.token_env).strip():
        raise ValueError("git_auth.token_env is required")
    if cfg.git_auth.fallback_token_env is not None:
        cfg.git_auth.fallback_token_env = str(cfg.git_auth.fallback_token_env).strip() or None

    mode = str(cfg.runner.mode).strip().lower()
    if mode not in {"local", "ssh"}:
        raise ValueError("runner.mode must be one of: local, ssh")
    cfg.runner.mode = mode
    if cfg.runner.local.activation_command is not None:
        cfg.runner.local.activation_command = str(cfg.runner.local.activation_command).strip() or None
    if cfg.runner.mode == "ssh":
        if not cfg.runner.ssh.host_alias and not cfg.runner.ssh.host:
            raise ValueError("runner.ssh.host_alias or runner.ssh.host is required when runner.mode = 'ssh'")
        if cfg.runner.ssh.port <= 0:
            raise ValueError("runner.ssh.port must be > 0")
        if not str(cfg.runner.ssh.repo_path).strip():
            raise ValueError("runner.ssh.repo_path is required when runner.mode = 'ssh'")
        if not str(cfg.runner.ssh.git_remote).strip():
            raise ValueError("runner.ssh.git_remote is required when runner.mode = 'ssh'")
        if not str(cfg.runner.ssh.runtime_root).strip():
            raise ValueError("runner.ssh.runtime_root is required when runner.mode = 'ssh'")
    if cfg.runner.ssh.activation_command is not None:
        cfg.runner.ssh.activation_command = str(cfg.runner.ssh.activation_command).strip() or None

    if not cfg.runner.scout.commands:
        raise ValueError("runner.scout.commands must not be empty")
    for stage_name in STAGE_NAMES:
        stage = getattr(cfg.runner, stage_name)
        if stage.enabled and not stage.commands:
            raise ValueError(f"runner.{stage_name}.commands must not be empty when enabled=true")
        if stage.timeout_seconds < 1:
            raise ValueError(f"runner.{stage_name}.timeout_seconds must be >= 1")

    metric_names: set[str] = set()
    if not cfg.metrics:
        raise ValueError("At least one [[metrics]] entry is required")
    for metric in cfg.metrics:
        name = str(metric.name).strip()
        if not name:
            raise ValueError("Each metric.name is required")
        metric.name = name
        if name in metric_names:
            raise ValueError(f"Duplicate metric name: {name}")
        metric_names.add(name)
        source = str(metric.source).strip().lower()
        if source not in {"json_file", "regex", "metric_file"}:
            raise ValueError(f"Unsupported metric source for {name}: {metric.source}")
        metric.source = source
        if source in {"json_file", "metric_file"} and not str(metric.path or "").strip():
            raise ValueError(f"metrics.{name}.path is required for {source}")
        if source == "json_file" and not str(metric.json_key or "").strip():
            raise ValueError(f"metrics.{name}.json_key is required for json_file metrics")
        if source == "regex" and not str(metric.pattern or "").strip():
            raise ValueError(f"metrics.{name}.pattern is required for regex metrics")

    primary_metric = str(cfg.objective.primary_metric).strip()
    if not primary_metric:
        raise ValueError("objective.primary_metric is required")
    if primary_metric not in metric_names:
        raise ValueError(f"objective.primary_metric references unknown metric: {primary_metric}")
    cfg.objective.primary_metric = primary_metric
    direction = str(cfg.objective.direction).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("objective.direction must be maximize or minimize")
    cfg.objective.direction = direction
    if cfg.objective.cost_penalty_metric and cfg.objective.cost_penalty_metric not in metric_names:
        raise ValueError(
            f"objective.cost_penalty_metric references unknown metric: {cfg.objective.cost_penalty_metric}"
        )
    if cfg.objective.cost_penalty_weight < 0.0:
        raise ValueError("objective.cost_penalty_weight must be >= 0")
    if cfg.objective.minimum_improvement_delta < 0.0:
        raise ValueError("objective.minimum_improvement_delta must be >= 0")

    policy = str(cfg.search.policy).strip().lower()
    if policy not in {"random", "greedy_best_first", "local_ranking"}:
        raise ValueError("search.policy must be one of: random, greedy_best_first, local_ranking")
    cfg.search.policy = policy
    if cfg.search.max_history < 1:
        raise ValueError("search.max_history must be >= 1")

    if not cfg.boa_md_path.exists():
        raise FileNotFoundError(f"Target repo is missing boa.md: {cfg.boa_md_path}")

    return cfg


def load_config(repo: str | Path, config_path: str | Path | None = None) -> BoaConfig:
    repo_root = find_repo_root(Path(repo))
    resolved_config_path = Path(config_path) if config_path is not None else (repo_root / "boa.config")
    if not resolved_config_path.is_absolute():
        resolved_config_path = (repo_root / resolved_config_path).resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(resolved_config_path)
    boa_md_path = (repo_root / "boa.md").resolve()
    raw = _load_toml_dict(resolved_config_path)
    cfg = _build_config(raw, repo_root=repo_root, config_path=resolved_config_path, boa_md_path=boa_md_path)
    return validate_config(cfg)
