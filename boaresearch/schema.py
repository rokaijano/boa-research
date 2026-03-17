from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


CONFIG_SCHEMA_VERSION = 2
StageName = Literal["scout", "confirm", "promoted"]
MetricSource = Literal["json_file", "regex", "metric_file"]
ObjectiveDirection = Literal["maximize", "minimize"]
AgentRuntime = Literal["cli", "deepagents"]
RunnerMode = Literal["local", "ssh"]

PATCH_CATEGORIES = {
    "optimizer",
    "architecture",
    "regularization",
    "data",
    "training_loop",
    "eval",
    "misc",
}
OPERATION_TYPES = {"add", "delete", "replace", "refactor"}


@dataclass
class RunConfig:
    tag: str = "default"
    max_trials: int = 1
    max_consecutive_failures: int = 3
    accepted_branch: Optional[str] = None
    base_branch: Optional[str] = None
    worktree_path: Optional[Path] = None
    stop_file: Optional[Path] = None


@dataclass
class AgentConfig:
    preset: str = "codex"
    runtime: str = "cli"
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    profile: Optional[str] = None
    prepare_timeout_seconds: int = 900
    model: str = ""
    reasoning_effort: Optional[str] = None
    max_output_tokens: int = 4000
    max_agent_steps: int = 40
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: Optional[str] = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    extra_context_files: list[Path] = field(default_factory=list)


@dataclass
class GuardrailConfig:
    allowed_paths: list[str] = field(default_factory=list)
    protected_paths: list[str] = field(default_factory=list)
    preflight_commands: list[str] = field(default_factory=list)


@dataclass
class GitAuthConfig:
    token_env: str = "GIT_TOKEN"
    fallback_token_env: Optional[str] = "GITHUB_TOKEN"
    username: str = "x-access-token"
    use_for_local_push: bool = True
    use_for_remote_fetch: bool = True
    persist_credentials: bool = False


@dataclass
class LocalRunnerConfig:
    activation_command: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SSHRunnerConfig:
    host: Optional[str] = None
    user: Optional[str] = None
    port: int = 22
    key_path: Optional[Path] = None
    host_alias: Optional[str] = None
    repo_path: str = "."
    git_remote: str = "origin"
    activation_command: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    runtime_root: str = ".boa/remote"


@dataclass
class RunnerStageConfig:
    enabled: bool = True
    commands: list[str] = field(default_factory=list)
    timeout_seconds: int = 1800
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    mode: str = "local"
    local: LocalRunnerConfig = field(default_factory=LocalRunnerConfig)
    ssh: SSHRunnerConfig = field(default_factory=SSHRunnerConfig)
    scout: RunnerStageConfig = field(default_factory=RunnerStageConfig)
    confirm: RunnerStageConfig = field(
        default_factory=lambda: RunnerStageConfig(enabled=False, commands=[], timeout_seconds=3600, env={})
    )
    promoted: RunnerStageConfig = field(
        default_factory=lambda: RunnerStageConfig(enabled=False, commands=[], timeout_seconds=7200, env={})
    )


@dataclass
class MetricConfig:
    name: str = ""
    source: str = "regex"
    path: Optional[str] = None
    json_key: Optional[str] = None
    pattern: Optional[str] = None
    group: str = "1"
    target: str = "combined"
    required: bool = True


@dataclass
class ObjectiveConfig:
    primary_metric: str = ""
    direction: str = "maximize"
    threshold: Optional[float] = None
    cost_penalty_metric: Optional[str] = None
    cost_penalty_weight: float = 0.0
    minimum_improvement_delta: float = 0.0


@dataclass
class SearchConfig:
    policy: str = "random"
    seed: Optional[int] = None
    max_history: int = 50
    risk_penalty: float = 0.25
    family_bonus: float = 0.1
    lineage_bonus: float = 0.05


@dataclass
class BoaConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    run: RunConfig = field(default_factory=RunConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    git_auth: GitAuthConfig = field(default_factory=GitAuthConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    metrics: list[MetricConfig] = field(default_factory=list)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    repo_root: Path = Path(".")
    config_path: Path = Path("boa.config")
    boa_md_path: Path = Path("boa.md")


@dataclass
class SearchDecision:
    policy: str
    parent_branch: str
    parent_trial_id: Optional[str] = None
    patch_category_hint: Optional[str] = None
    prompt_hints: list[str] = field(default_factory=list)
    budget_hint: Optional[str] = None


@dataclass
class CandidateMetadata:
    hypothesis: str
    rationale_summary: str
    patch_category: str
    operation_type: str
    estimated_risk: float
    target_symbols: list[str] = field(default_factory=list)
    numeric_knobs: dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None


@dataclass
class PatchDescriptor:
    touched_files: list[str]
    touched_symbols: list[str]
    patch_category: str
    operation_type: str
    numeric_knobs: dict[str, float]
    rationale_summary: str
    estimated_risk: float
    parent_branch: str
    parent_trial_id: Optional[str]
    budget_used: str
    diff_path: str


@dataclass
class TrialSummary:
    trial_id: str
    branch_name: str
    parent_branch: str
    parent_trial_id: Optional[str]
    acceptance_status: str
    canonical_stage: Optional[str]
    canonical_score: Optional[float]
    candidate: Optional[CandidateMetadata]
    descriptor: Optional[PatchDescriptor]
    stage_scores: dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class IncumbentRecord:
    stage_name: str
    trial_id: str
    branch_name: str
    adjusted_score: float
    primary_metric: float
    updated_at: str


@dataclass
class StageCommandResult:
    index: int
    command: str
    exit_code: int
    status: str
    stdout_path: Path
    stderr_path: Path
    wall_time_seconds: float
    max_rss_kb: Optional[int] = None


@dataclass
class StageRunResult:
    stage_name: str
    branch_name: str
    status: str
    command_results: list[StageCommandResult]
    metrics: dict[str, float]
    primary_metric: Optional[float]
    cost_metric: Optional[float]
    adjusted_score: Optional[float]
    threshold_passed: bool
    improved: bool
    advanced: bool
    final_accept: bool
    reason: str
    started_at: str
    completed_at: str
    resource_metadata: dict[str, Any]
    artifact_dir: Path


@dataclass
class StageEvaluation:
    stage_name: str
    primary_metric: float
    adjusted_score: float
    threshold_passed: bool
    improved: bool
    advanced: bool
    final_accept: bool
    reason: str
    cost_metric: Optional[float] = None


@dataclass
class AgentContext:
    repo_root: Path
    worktree_path: Path
    trial_id: str
    run_tag: str
    accepted_branch: str
    trial_branch: str
    boa_md_path: Path
    extra_context_files: list[Path]
    allowed_paths: list[str]
    protected_paths: list[str]
    search_decision: SearchDecision
    recent_trials: list[TrialSummary]
    objective_summary: str
    preflight_commands: list[str]
    max_agent_steps: int
    prompt_bundle_dir: Path
