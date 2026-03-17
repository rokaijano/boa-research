from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .schema import BoaConfig


ExistingSetupStatus = Literal["absent", "valid", "invalid"]
ExistingSetupAction = Literal["review", "overwrite", "update"]


@dataclass
class ExistingSetupReport:
    status: str = "absent"
    config_path: Path | None = None
    boa_md_path: Path | None = None
    config: BoaConfig | None = None
    summary: str = ""
    raw_preview: str = ""


@dataclass
class DetectedRepo:
    requested_path: Path
    repo_root: Path
    config_path: Path
    boa_md_path: Path
    runtime_root: Path
    existing_setup: ExistingSetupReport = field(default_factory=ExistingSetupReport)


@dataclass
class InitSetupSelection:
    repo_root: Path
    existing_action: str = "overwrite"
    agent_preset: str = "codex"
    agent_runtime: str = "cli"
    agent_command: str = "codex"
    agent_args: list[str] = field(default_factory=list)
    agent_env: dict[str, str] = field(default_factory=dict)
    agent_profile: Optional[str] = None
    agent_model: str = ""
    agent_reasoning_effort: Optional[str] = None
    agent_backend: str = "ollama"
    agent_base_url: str = "http://127.0.0.1:11434"
    agent_api_key_env: Optional[str] = None
    runner_mode: str = "local"
    local_activation_command: Optional[str] = None
    local_env: dict[str, str] = field(default_factory=dict)
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    ssh_key_path: Optional[Path] = None
    ssh_host_alias: Optional[str] = None
    ssh_repo_path: str = "."
    ssh_git_remote: str = "origin"
    ssh_activation_command: Optional[str] = None
    ssh_env: dict[str, str] = field(default_factory=dict)


@dataclass
class RepoAnalysisProposal:
    train_command: str
    eval_command: str
    primary_metric_name: str
    metric_direction: str
    metric_source: str
    metric_path: Optional[str]
    metric_json_key: Optional[str]
    metric_pattern: Optional[str]
    editable_files: list[str] = field(default_factory=list)
    protected_files: list[str] = field(default_factory=list)
    optimization_surfaces: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    suggested_boa_md: str = ""


@dataclass
class ReviewedInitPlan:
    repo_root: Path
    selection: InitSetupSelection
    analysis: RepoAnalysisProposal
    editable_files: list[str]
    protected_files: list[str]
    train_command: str
    eval_command: str
    primary_metric_name: str
    metric_direction: str
    metric_source: str
    metric_path: Optional[str]
    metric_json_key: Optional[str]
    metric_pattern: Optional[str]
    boa_md: str


@dataclass
class PreflightCheck:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass
class WriteResult:
    created_paths: list[Path] = field(default_factory=list)
    updated_paths: list[Path] = field(default_factory=list)
    skipped_paths: list[Path] = field(default_factory=list)


@dataclass
class ValidationReport:
    passed: bool
    details: list[str] = field(default_factory=list)


@dataclass
class InitDraft:
    detected_repo: DetectedRepo | None = None
    selection: InitSetupSelection | None = None
    preflight_checks: list[PreflightCheck] = field(default_factory=list)
    analysis: RepoAnalysisProposal | None = None
    reviewed_plan: ReviewedInitPlan | None = None
    write_result: WriteResult | None = None
    validation: ValidationReport | None = None
