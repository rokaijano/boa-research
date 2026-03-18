from __future__ import annotations

import copy
import importlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..agent_presets import resolve_agent_profile
from ..agents.base import ResearchAgentError, extract_json_object
from ..agents.deepagents import _call_with_supported_kwargs, _import_attr
from ..prompt_templates import read_prompt_template, render_prompt_template
from .models import (
    DetectedRepo,
    ExistingSetupReport,
    InitSetupSelection,
    PreflightCheck,
    RepoAnalysisProposal,
    ReviewedInitPlan,
    ValidationReport,
    WriteResult,
)
from ..loader import find_repo_root, load_config
from ..runner import build_trial_runner
from ..schema import (
    AgentConfig,
    BoaConfig,
    CONFIG_SCHEMA_VERSION,
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


REPO_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "train_command",
        "eval_command",
        "primary_metric_name",
        "metric_direction",
        "metric_source",
        "metric_path",
        "metric_json_key",
        "metric_pattern",
        "editable_files",
        "protected_files",
        "optimization_surfaces",
        "caveats",
        "suggested_boa_md",
    ],
    "properties": {
        "train_command": {"type": "string"},
        "eval_command": {"type": "string"},
        "primary_metric_name": {"type": "string"},
        "metric_direction": {"type": "string", "enum": ["maximize", "minimize"]},
        "metric_source": {"type": "string", "enum": ["json_file", "regex", "metric_file"]},
        "metric_path": {"type": ["string", "null"]},
        "metric_json_key": {"type": ["string", "null"]},
        "metric_pattern": {"type": ["string", "null"]},
        "editable_files": {"type": "array", "items": {"type": "string"}},
        "protected_files": {"type": "array", "items": {"type": "string"}},
        "optimization_surfaces": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "suggested_boa_md": {"type": "string"},
    },
}


def _shell_args(command: str) -> list[str]:
    return ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["bash", "-lc", command]


def _preview_text(path: Path, *, limit: int = 1600) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated] ..."


def _first_existing(repo_root: Path, candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if (repo_root / candidate).exists():
            return candidate
    return None


def _normalize_paths(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw in values:
        cleaned = str(raw).replace("\\", "/").strip()
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        cleaned = cleaned.rstrip("/")
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _looks_like_data_path(path: str) -> bool:
    normalized = _normalize_paths([path])
    if not normalized:
        return False
    value = normalized[0]
    parts = [part.lower() for part in value.split("/") if part]
    if not parts:
        return False
    data_names = {
        "data",
        "dataset",
        "datasets",
        "corpus",
        "corpora",
        "raw_data",
        "processed_data",
        "sample_data",
        "fixtures_data",
    }
    if any(part in data_names for part in parts):
        return True
    suffix = Path(value).suffix.lower()
    return suffix in {
        ".csv",
        ".tsv",
        ".jsonl",
        ".parquet",
        ".feather",
        ".arrow",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".ckpt",
        ".h5",
        ".hdf5",
        ".pkl",
        ".pickle",
        ".joblib",
        ".onnx",
        ".bin",
        ".gz",
        ".bz2",
        ".xz",
        ".zip",
        ".tar",
        ".tgz",
    }


def _looks_like_eval_path(path: str) -> bool:
    normalized = _normalize_paths([path])
    if not normalized:
        return False
    value = normalized[0]
    name = Path(value).name.lower()
    eval_names = {
        "eval.py",
        "evaluate.py",
        "evaluation.py",
        "eval.sh",
        "evaluate.sh",
        "evaluation.sh",
    }
    return name in eval_names


def _is_sensitive_path(path: str) -> bool:
    normalized = _normalize_paths([path])
    if not normalized:
        return False
    value = normalized[0]
    return value in {".boa", ".git"} or _looks_like_eval_path(value) or _looks_like_data_path(value)


def _default_editable_paths(repo_root: Path) -> list[str]:
    editable: list[str] = []
    for top_level in ("src", "app"):
        if (repo_root / top_level).exists():
            editable.append(top_level)
    if editable:
        return editable
    for candidate in sorted(repo_root.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name in {".git", ".boa", "tests", "docs", "reports", "artifacts", "build", "dist", "vendor", "node_modules"}:
            continue
        editable.append(candidate.name)
        if len(editable) >= 3:
            break
    if editable:
        return editable
    if (repo_root / "train").exists():
        return ["train"]
    return ["src"]


def _default_protected_paths(repo_root: Path, *, metric_path: str | None) -> list[str]:
    protected = [".boa", ".git"]
    for candidate in ("eval.py", "evaluate.py", "evaluation.py", "src/eval.py", "src/evaluate.py", "src/evaluation.py"):
        if (repo_root / candidate).exists():
            protected.append(candidate)
    for candidate in sorted(repo_root.iterdir()):
        if not candidate.exists():
            continue
        rel = candidate.relative_to(repo_root).as_posix()
        if _looks_like_data_path(rel):
            protected.append(rel)
    if metric_path:
        protected.append(metric_path)
    return _normalize_paths(protected)


def _normalize_aggressiveness(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"light", "normal", "aggressive"}:
        return normalized
    return "normal"


def _aggressiveness_prompt(mode: str) -> tuple[str, list[str]]:
    normalized = _normalize_aggressiveness(mode)
    raw = read_prompt_template("init", "aggressiveness", f"{normalized}.md")
    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
    summary = lines[0].strip()
    rules = [line.lstrip("-").strip() for line in lines[1:]]
    return summary, rules


def _aggressiveness_summary(mode: str) -> str:
    summary, _rules = _aggressiveness_prompt(mode)
    return summary


def _aggressiveness_rules(mode: str) -> list[str]:
    _summary, rules = _aggressiveness_prompt(mode)
    return rules


def _parse_assignment_map(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _quote(value: str) -> str:
    return json.dumps(str(value))


def _render_list(values: list[str]) -> str:
    return "[" + ", ".join(_quote(value) for value in values) + "]"


def _render_str_map(values: dict[str, str]) -> list[str]:
    return [f"{key} = {_quote(value)}" for key, value in sorted(values.items())]


def summarize_config(config: BoaConfig) -> str:
    lines = [
        f"Repo: {config.repo_root}",
        f"Agent: {config.agent.preset} ({config.agent.runtime})",
        f"Runner: {config.runner.mode}",
        f"Primary metric: {config.objective.primary_metric} ({config.objective.direction})",
        f"Editable paths: {', '.join(config.guardrails.allowed_paths) if config.guardrails.allowed_paths else '<all tracked>'}",
        f"Protected paths: {', '.join(config.guardrails.protected_paths) if config.guardrails.protected_paths else '<none>'}",
    ]
    if config.runner.mode == "ssh":
        lines.append(
            "SSH target: "
            + (config.runner.ssh.host_alias or f"{config.runner.ssh.user or ''}@{config.runner.ssh.host}".strip("@"))
        )
    return "\n".join(lines)


def detect_repo(start_path: Path) -> DetectedRepo:
    requested = start_path.resolve()
    probe = requested if requested.is_dir() else requested.parent
    try:
        repo_root = find_repo_root(probe)
    except FileNotFoundError as exc:
        raise ValueError(f"{requested} is not inside a Git repository") from exc
    detected = DetectedRepo(
        requested_path=requested,
        repo_root=repo_root,
        config_path=(repo_root / "boa.config").resolve(),
        boa_md_path=(repo_root / "boa.md").resolve(),
        runtime_root=(repo_root / ".boa").resolve(),
    )
    detected.existing_setup = inspect_existing_setup(detected)
    return detected


def inspect_existing_setup(detected: DetectedRepo) -> ExistingSetupReport:
    config_exists = detected.config_path.exists()
    boa_md_exists = detected.boa_md_path.exists()
    if not config_exists and not boa_md_exists:
        return ExistingSetupReport(status="absent", config_path=detected.config_path, boa_md_path=detected.boa_md_path)
    try:
        config = load_config(detected.repo_root, detected.config_path)
    except Exception:
        preview_blocks: list[str] = []
        if config_exists:
            preview_blocks.append(f"boa.config\n{_preview_text(detected.config_path)}")
        if boa_md_exists:
            preview_blocks.append(f"boa.md\n{_preview_text(detected.boa_md_path)}")
        return ExistingSetupReport(
            status="invalid",
            config_path=detected.config_path,
            boa_md_path=detected.boa_md_path,
            raw_preview="\n\n".join(preview_blocks),
        )
    return ExistingSetupReport(
        status="valid",
        config_path=detected.config_path,
        boa_md_path=detected.boa_md_path,
        config=config,
        summary=summarize_config(config),
    )


def default_selection_for_repo(detected: DetectedRepo) -> InitSetupSelection:
    selection = InitSetupSelection(repo_root=detected.repo_root)
    if detected.existing_setup.status == "valid" and detected.existing_setup.config is not None:
        config = detected.existing_setup.config
        selection.existing_config = copy.deepcopy(config)
        selection.existing_action = "update"
        selection.agent_preset = config.agent.preset
        selection.agent_runtime = config.agent.runtime
        selection.agent_command = str(config.agent.command or "")
        selection.agent_args = list(config.agent.args)
        selection.agent_env = dict(config.agent.env)
        selection.agent_profile = config.agent.profile
        selection.agent_model = config.agent.model
        selection.agent_reasoning_effort = config.agent.reasoning_effort
        selection.agent_backend = config.agent.backend
        selection.agent_base_url = config.agent.base_url
        selection.agent_api_key_env = config.agent.api_key_env
        selection.runner_mode = config.runner.mode
        selection.local_activation_command = config.runner.local.activation_command
        selection.local_env = dict(config.runner.local.env)
        selection.ssh_host = config.runner.ssh.host
        selection.ssh_user = config.runner.ssh.user
        selection.ssh_port = config.runner.ssh.port
        selection.ssh_key_path = config.runner.ssh.key_path
        selection.ssh_host_alias = config.runner.ssh.host_alias
        selection.ssh_repo_path = config.runner.ssh.repo_path
        selection.ssh_git_remote = config.runner.ssh.git_remote
        selection.ssh_activation_command = config.runner.ssh.activation_command
        selection.ssh_env = dict(config.runner.ssh.env)
        return selection
    agent_defaults = resolve_agent_profile({"preset": "codex"})
    selection.agent_command = str(agent_defaults.get("command") or "codex")
    selection.agent_runtime = str(agent_defaults.get("runtime") or "cli")
    return selection


def resolve_agent_config(selection: InitSetupSelection) -> AgentConfig:
    raw = resolve_agent_profile(
        {
            "preset": selection.agent_preset,
            "runtime": selection.agent_runtime,
            "command": selection.agent_command or None,
            "args": list(selection.agent_args),
            "env": dict(selection.agent_env),
            "profile": selection.agent_profile,
            "model": selection.agent_model,
            "reasoning_effort": selection.agent_reasoning_effort,
            "backend": selection.agent_backend,
            "base_url": selection.agent_base_url,
            "api_key_env": selection.agent_api_key_env,
        }
    )
    return AgentConfig(**raw)


def build_runner_config(selection: InitSetupSelection, *, scout_commands: list[str]) -> RunnerConfig:
    return RunnerConfig(
        mode=selection.runner_mode,
        local=LocalRunnerConfig(
            activation_command=selection.local_activation_command,
            env=dict(selection.local_env),
        ),
        ssh=SSHRunnerConfig(
            host=selection.ssh_host,
            user=selection.ssh_user,
            port=int(selection.ssh_port),
            key_path=selection.ssh_key_path,
            host_alias=selection.ssh_host_alias,
            repo_path=selection.ssh_repo_path,
            git_remote=selection.ssh_git_remote,
            activation_command=selection.ssh_activation_command,
            env=dict(selection.ssh_env),
        ),
        scout=RunnerStageConfig(commands=scout_commands),
    )


def run_preflight(detected: DetectedRepo, selection: InitSetupSelection) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    git_ok = shutil.which("git") is not None
    checks.append(PreflightCheck("git", git_ok, "git found" if git_ok else "git is not installed"))

    repo_writable = os.access(detected.repo_root, os.W_OK)
    checks.append(
        PreflightCheck(
            "repo_write",
            repo_writable,
            f"{detected.repo_root} is writable" if repo_writable else f"{detected.repo_root} is not writable",
        )
    )

    try:
        detected.runtime_root.mkdir(parents=True, exist_ok=True)
        boa_dir_ok = True
        boa_detail = f"{detected.runtime_root} is available"
    except OSError as exc:
        boa_dir_ok = False
        boa_detail = f"Unable to create {detected.runtime_root}: {exc}"
    checks.append(PreflightCheck("boa_dir", boa_dir_ok, boa_detail))

    agent = resolve_agent_config(selection)
    if agent.runtime == "cli":
        executable = shlex.split(agent.command or "")[:1]
        binary = executable[0] if executable else ""
        agent_ok = bool(binary) and shutil.which(binary) is not None
        checks.append(
            PreflightCheck(
                "agent",
                agent_ok,
                f"CLI agent '{binary}' is callable" if agent_ok else f"CLI agent '{binary or '<empty>'}' was not found",
            )
        )
    else:
        deepagents_ok = importlib.util.find_spec("deepagents") is not None
        detail = "deepagents import is available" if deepagents_ok else "deepagents is not installed"
        if deepagents_ok and agent.backend == "openai":
            key_name = agent.api_key_env or "OPENAI_API_KEY"
            api_ok = bool(os.environ.get(key_name))
            checks.append(
                PreflightCheck(
                    "agent_backend",
                    api_ok,
                    f"{key_name} is set" if api_ok else f"{key_name} is required for openai backend",
                )
            )
        checks.append(PreflightCheck("agent", deepagents_ok, detail))

    if selection.runner_mode == "local":
        checks.append(PreflightCheck("runner", True, "Local execution selected"))
    else:
        ssh_ok = shutil.which("ssh") is not None
        target_ok = bool(selection.ssh_host_alias or selection.ssh_host)
        repo_path_ok = bool(str(selection.ssh_repo_path).strip())
        key_ok = selection.ssh_key_path is None or selection.ssh_key_path.exists()
        checks.extend(
            [
                PreflightCheck("ssh_binary", ssh_ok, "ssh found" if ssh_ok else "ssh is not installed"),
                PreflightCheck(
                    "ssh_target",
                    target_ok,
                    "SSH target configured" if target_ok else "Provide runner.ssh host or host alias",
                ),
                PreflightCheck(
                    "ssh_repo_path",
                    repo_path_ok,
                    "Remote repo path configured" if repo_path_ok else "Provide a remote repo path",
                ),
                PreflightCheck(
                    "ssh_key",
                    key_ok,
                    "SSH key path looks valid" if key_ok else f"SSH key not found: {selection.ssh_key_path}",
                ),
            ]
        )
    return checks


def _tree_listing(repo_root: Path, *, limit: int = 250) -> str:
    lines: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if ".git" in path.parts or ".boa" in path.parts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        if path.is_dir():
            rel += "/"
        lines.append(rel)
        if len(lines) >= limit:
            lines.append("... [truncated]")
            break
    return "\n".join(lines)


def _likely_context_files(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("README*", "*.md", "pyproject.toml", "requirements*.txt", "setup.py", "Makefile", "*.yaml", "*.yml"):
        for match in sorted(repo_root.glob(pattern)):
            if match.is_file() and match not in candidates:
                candidates.append(match)
    for name in (
        "train.py",
        "eval.py",
        "evaluate.py",
        "main.py",
        "src/train.py",
        "src/eval.py",
        "tests/test_train.py",
    ):
        path = repo_root / name
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates[:12]


def default_repo_analysis(repo_root: Path) -> RepoAnalysisProposal:
    train_file = _first_existing(repo_root, ["train.py", "src/train.py", "main.py"])
    eval_file = _first_existing(repo_root, ["eval.py", "evaluate.py", "src/eval.py"])
    metric_json = _first_existing(repo_root, ["reports/metrics.json", "metrics.json", "artifacts/metrics.json"])
    editable = _default_editable_paths(repo_root)
    protected = _default_protected_paths(repo_root, metric_path=metric_json)
    optimization_surfaces = [item for item in editable if item not in {"tests"}]
    caveats: list[str] = []
    if train_file is None:
        caveats.append("No obvious train entrypoint detected.")
    if eval_file is None:
        caveats.append("No obvious eval entrypoint detected.")
    if metric_json is None:
        caveats.append("Metric extraction will need manual confirmation.")
    train_command = f"python {train_file}" if train_file else "python train.py"
    eval_command = f"python {eval_file}" if eval_file else "python eval.py"
    metric_name = "accuracy"
    metric_source = "json_file" if metric_json else "regex"
    metric_path = metric_json
    metric_json_key = "accuracy" if metric_json else None
    metric_pattern = None if metric_json else "accuracy=([0-9.]+)"
    boa_md = "\n".join(
        [
            "# BOA Repo Contract",
            "",
            "Focus on the training and evaluation workflow for this repository.",
            f"Preferred editable paths: {', '.join(_normalize_paths(editable)) or '<confirm in review>'}.",
            f"Protected paths: {', '.join(_normalize_paths(protected))}.",
            "Keep changes coherent, measurable, and easy to evaluate with the configured stage commands.",
        ]
    )
    return RepoAnalysisProposal(
        train_command=train_command,
        eval_command=eval_command,
        primary_metric_name=metric_name,
        metric_direction="maximize",
        metric_source=metric_source,
        metric_path=metric_path,
        metric_json_key=metric_json_key,
        metric_pattern=metric_pattern,
        editable_files=_normalize_paths(editable),
        protected_files=_normalize_paths(protected),
        optimization_surfaces=_normalize_paths(optimization_surfaces),
        caveats=caveats,
        suggested_boa_md=boa_md,
    )


def _analysis_prompt(repo_root: Path, selection: InitSetupSelection) -> str:
    context_files = _likely_context_files(repo_root)
    excerpt_blocks: list[str] = []
    for path in context_files:
        rel = path.relative_to(repo_root).as_posix()
        excerpt_blocks.append(f"FILE: {rel}\n{_preview_text(path, limit=4000)}")
    return render_prompt_template(
        "init",
        "repo_analysis.md",
        repo_root=repo_root,
        agent_preset=selection.agent_preset,
        runner_mode=selection.runner_mode,
        agent_aggressiveness=_normalize_aggressiveness(selection.agent_aggressiveness),
        aggressiveness_summary=_aggressiveness_summary(selection.agent_aggressiveness),
        repository_tree=_tree_listing(repo_root),
        context_file_excerpts="\n\n".join(excerpt_blocks) if excerpt_blocks else "<no context file excerpts>",
        repo_analysis_schema=json.dumps(REPO_ANALYSIS_SCHEMA, indent=2, sort_keys=True),
    )


def parse_repo_analysis(text: str) -> RepoAnalysisProposal:
    data = extract_json_object(text)
    required = [
        "train_command",
        "eval_command",
        "primary_metric_name",
        "metric_direction",
        "metric_source",
        "editable_files",
        "protected_files",
        "optimization_surfaces",
        "caveats",
        "suggested_boa_md",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ResearchAgentError(f"Analysis output missing required fields: {', '.join(missing)}")
    direction = str(data["metric_direction"]).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ResearchAgentError("metric_direction must be maximize or minimize")
    source = str(data["metric_source"]).strip().lower()
    if source not in {"json_file", "regex", "metric_file"}:
        raise ResearchAgentError("metric_source must be json_file, regex, or metric_file")
    return RepoAnalysisProposal(
        train_command=str(data["train_command"]).strip(),
        eval_command=str(data["eval_command"]).strip(),
        primary_metric_name=str(data["primary_metric_name"]).strip(),
        metric_direction=direction,
        metric_source=source,
        metric_path=None if data.get("metric_path") is None else str(data.get("metric_path")).strip() or None,
        metric_json_key=None if data.get("metric_json_key") is None else str(data.get("metric_json_key")).strip() or None,
        metric_pattern=None if data.get("metric_pattern") is None else str(data.get("metric_pattern")).strip() or None,
        editable_files=_normalize_paths([str(item) for item in list(data.get("editable_files") or [])]),
        protected_files=_normalize_paths([str(item) for item in list(data.get("protected_files") or [])]),
        optimization_surfaces=_normalize_paths([str(item) for item in list(data.get("optimization_surfaces") or [])]),
        caveats=[str(item).strip() for item in list(data.get("caveats") or []) if str(item).strip()],
        suggested_boa_md=str(data["suggested_boa_md"]).strip(),
    )


def _run_cli_analysis(selection: InitSetupSelection, prompt: str, cwd: Path) -> RepoAnalysisProposal:
    agent = resolve_agent_config(selection)
    base_command = str(agent.command)
    env = os.environ.copy()
    env.update(agent.env)
    if agent.preset == "codex" and Path(base_command).name == "codex":
        schema_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        output_file = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        try:
            schema_file.write(json.dumps(REPO_ANALYSIS_SCHEMA, indent=2, sort_keys=True))
            schema_file.flush()
            output_file.close()
            command = [
                base_command,
                "exec",
                prompt,
                "--color",
                "never",
                "--output-schema",
                schema_file.name,
                "--output-last-message",
                output_file.name,
                "-C",
                str(cwd),
            ]
            if agent.profile:
                command.extend(["-p", str(agent.profile)])
            if agent.model:
                command.extend(["-m", str(agent.model)])
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
                timeout=int(agent.prepare_timeout_seconds),
                env=env,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise ResearchAgentError(f"CLI analysis agent exited with code {proc.returncode}: {detail}")
            return parse_repo_analysis(Path(output_file.name).read_text(encoding="utf-8"))
        finally:
            Path(schema_file.name).unlink(missing_ok=True)
            Path(output_file.name).unlink(missing_ok=True)
    command = [base_command, *list(agent.args)]
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
        timeout=int(agent.prepare_timeout_seconds),
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ResearchAgentError(f"CLI analysis agent exited with code {proc.returncode}: {detail}")
    return parse_repo_analysis(proc.stdout or "")


def _run_deepagents_analysis(selection: InitSetupSelection, prompt: str) -> RepoAnalysisProposal:
    agent = resolve_agent_config(selection)
    init_chat_model = _import_attr(["langchain.chat_models", "langchain.chat_models.base"], "init_chat_model")
    model_name = str(agent.model).strip()
    if model_name and not model_name.startswith("openai:") and not model_name.startswith("ollama:"):
        model_name = f"{agent.backend}:{model_name}"
    kwargs = {
        "base_url": str(agent.base_url).strip(),
        "max_tokens": int(agent.max_output_tokens),
        "max_output_tokens": int(agent.max_output_tokens),
    }
    if agent.reasoning_effort:
        kwargs["reasoning_effort"] = str(agent.reasoning_effort)
    kwargs.update(dict(agent.provider_options))
    if agent.backend == "openai":
        api_key_env = agent.api_key_env or "OPENAI_API_KEY"
        api_key = str(os.environ.get(api_key_env, "")).strip()
        if not api_key:
            raise ResearchAgentError(f"Missing API key environment variable: {api_key_env}")
        kwargs["api_key"] = api_key
    model = _call_with_supported_kwargs(init_chat_model, model_name, **kwargs)
    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        rendered = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    else:
        rendered = str(content)
    return parse_repo_analysis(rendered)


def analyze_repo(detected: DetectedRepo, selection: InitSetupSelection) -> RepoAnalysisProposal:
    prompt = _analysis_prompt(detected.repo_root, selection)
    agent = resolve_agent_config(selection)
    if agent.runtime == "cli":
        return _run_cli_analysis(selection, prompt, detected.repo_root)
    if agent.runtime == "deepagents":
        return _run_deepagents_analysis(selection, prompt)
    raise ResearchAgentError(f"Unsupported agent.runtime: {agent.runtime}")


def merge_reviewed_plan(selection: InitSetupSelection, analysis: RepoAnalysisProposal) -> ReviewedInitPlan:
    protected_candidates = list(analysis.protected_files)
    metric_path = analysis.metric_path
    if metric_path:
        protected_candidates.append(metric_path)
    protected = _normalize_paths(
        list(
            {
                ".boa",
                ".git",
                *protected_candidates,
            }
        )
    )
    editable = []
    for path in _normalize_paths(analysis.editable_files or _default_editable_paths(selection.repo_root)):
        if _is_sensitive_path(path):
            if path not in protected:
                protected.append(path)
            continue
        editable.append(path)
    if not editable:
        editable = [path for path in _default_editable_paths(selection.repo_root) if path not in protected]
    boa_md = analysis.suggested_boa_md.strip() or default_repo_analysis(selection.repo_root).suggested_boa_md
    return ReviewedInitPlan(
        repo_root=selection.repo_root,
        selection=selection,
        analysis=analysis,
        editable_files=editable,
        protected_files=protected,
        train_command=analysis.train_command.strip(),
        eval_command=analysis.eval_command.strip(),
        primary_metric_name=analysis.primary_metric_name.strip(),
        metric_direction=analysis.metric_direction.strip(),
        metric_source=analysis.metric_source.strip(),
        metric_path=analysis.metric_path,
        metric_json_key=analysis.metric_json_key,
        metric_pattern=analysis.metric_pattern,
        boa_md=boa_md,
    )


def build_config_from_plan(plan: ReviewedInitPlan) -> BoaConfig:
    selection = plan.selection
    metric = MetricConfig(
        name=plan.primary_metric_name,
        source=plan.metric_source,
        path=plan.metric_path,
        json_key=plan.metric_json_key,
        pattern=plan.metric_pattern,
    )
    base = None
    if selection.existing_action == "update" and selection.existing_config is not None:
        base = copy.deepcopy(selection.existing_config)
    if base is None:
        base = BoaConfig(
            schema_version=CONFIG_SCHEMA_VERSION,
            run=RunConfig(),
            agent=resolve_agent_config(selection),
            guardrails=GuardrailConfig(),
            runner=build_runner_config(selection, scout_commands=[plan.train_command, plan.eval_command]),
            metrics=[metric],
            objective=ObjectiveConfig(),
            search=SearchConfig(oracle="bayesian_optimization"),
            repo_root=plan.repo_root,
            config_path=(plan.repo_root / "boa.config"),
            boa_md_path=(plan.repo_root / "boa.md"),
        )
    else:
        updated_agent = resolve_agent_config(selection)
        base.agent.preset = updated_agent.preset
        base.agent.runtime = updated_agent.runtime
        base.agent.command = updated_agent.command
        base.agent.args = list(updated_agent.args)
        base.agent.env = dict(updated_agent.env)
        base.agent.profile = updated_agent.profile
        base.agent.model = updated_agent.model
        base.agent.reasoning_effort = updated_agent.reasoning_effort
        base.agent.backend = updated_agent.backend
        base.agent.base_url = updated_agent.base_url
        base.agent.api_key_env = updated_agent.api_key_env
        updated_runner = build_runner_config(selection, scout_commands=[plan.train_command, plan.eval_command])
        base.runner.mode = updated_runner.mode
        base.runner.local = updated_runner.local
        base.runner.ssh = updated_runner.ssh
        base.runner.scout.commands = list(updated_runner.scout.commands)
    base.schema_version = CONFIG_SCHEMA_VERSION
    base.repo_root = plan.repo_root
    base.config_path = plan.repo_root / "boa.config"
    base.boa_md_path = plan.repo_root / "boa.md"
    base.guardrails.allowed_paths = list(plan.editable_files)
    base.guardrails.protected_paths = list(plan.protected_files)
    merged_metrics: list[MetricConfig] = []
    replaced_metric = False
    for existing_metric in list(base.metrics):
        if existing_metric.name == metric.name:
            merged_metrics.append(copy.deepcopy(metric))
            replaced_metric = True
        else:
            merged_metrics.append(copy.deepcopy(existing_metric))
    if not replaced_metric:
        merged_metrics.insert(0, copy.deepcopy(metric))
    base.metrics = merged_metrics or [copy.deepcopy(metric)]
    base.objective.primary_metric = plan.primary_metric_name
    base.objective.direction = plan.metric_direction
    if not str(base.search.oracle).strip():
        base.search.oracle = "bayesian_optimization"
    return base


def _render_path_value(path: Path, repo_root: Path) -> str:
    try:
        value = str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        value = str(path)
    return _quote(value)


def _render_metric(metric: MetricConfig) -> list[str]:
    lines = [
        "[[metrics]]",
        f"name = {_quote(metric.name)}",
        f"source = {_quote(metric.source)}",
    ]
    if metric.path:
        lines.append(f"path = {_quote(metric.path)}")
    if metric.json_key:
        lines.append(f"json_key = {_quote(metric.json_key)}")
    if metric.pattern:
        lines.append(f"pattern = {_quote(metric.pattern)}")
    if metric.group != "1":
        lines.append(f"group = {_quote(metric.group)}")
    if metric.target != "combined":
        lines.append(f"target = {_quote(metric.target)}")
    if metric.required is not True:
        lines.append(f"required = {str(metric.required).lower()}")
    return lines


def render_config_text(plan: ReviewedInitPlan) -> str:
    config = build_config_from_plan(plan)
    lines: list[str] = [
        f"schema_version = {CONFIG_SCHEMA_VERSION}",
        "",
        "[run]",
        f"tag = {_quote(config.run.tag)}",
        f"max_trials = {config.run.max_trials}",
        f"max_consecutive_failures = {config.run.max_consecutive_failures}",
    ]
    if config.run.accepted_branch:
        lines.append(f"accepted_branch = {_quote(config.run.accepted_branch)}")
    if config.run.base_branch:
        lines.append(f"base_branch = {_quote(config.run.base_branch)}")
    if config.run.worktree_path:
        lines.append(f"worktree_path = {_render_path_value(config.run.worktree_path, config.repo_root)}")
    if config.run.stop_file:
        lines.append(f"stop_file = {_render_path_value(config.run.stop_file, config.repo_root)}")
    lines.extend(
        [
            "",
            "[agent]",
            f"preset = {_quote(config.agent.preset)}",
            f"runtime = {_quote(config.agent.runtime)}",
        ]
    )
    if config.agent.command:
        lines.append(f"command = {_quote(config.agent.command)}")
    if config.agent.args:
        lines.append(f"args = {_render_list(config.agent.args)}")
    if config.agent.profile:
        lines.append(f"profile = {_quote(config.agent.profile)}")
    if config.agent.model:
        lines.append(f"model = {_quote(config.agent.model)}")
    if config.agent.reasoning_effort:
        lines.append(f"reasoning_effort = {_quote(config.agent.reasoning_effort)}")
    lines.append(f"prepare_timeout_seconds = {config.agent.prepare_timeout_seconds}")
    lines.append(f"max_output_tokens = {config.agent.max_output_tokens}")
    lines.append(f"max_agent_steps = {config.agent.max_agent_steps}")
    if config.agent.runtime == "deepagents":
        lines.extend(
            [
                f"backend = {_quote(config.agent.backend)}",
                f"base_url = {_quote(config.agent.base_url)}",
            ]
        )
        if config.agent.api_key_env:
            lines.append(f"api_key_env = {_quote(config.agent.api_key_env)}")
    if config.agent.extra_context_files:
        extra_context = []
        for path in config.agent.extra_context_files:
            try:
                extra_context.append(str(path.resolve().relative_to(config.repo_root.resolve())).replace("\\", "/"))
            except ValueError:
                extra_context.append(str(path))
        lines.append(f"extra_context_files = {_render_list(extra_context)}")
    if config.agent.provider_options:
        lines.append("")
        lines.append("[agent.provider_options]")
        for key, value in sorted(config.agent.provider_options.items()):
            lines.append(f"{key} = {_quote(value)}")
    if config.agent.env:
        lines.append("")
        lines.append("[agent.env]")
        lines.extend(_render_str_map(config.agent.env))
    lines.extend(
        [
            "",
            "[guardrails]",
            f"allowed_paths = {_render_list(config.guardrails.allowed_paths)}",
            f"protected_paths = {_render_list(config.guardrails.protected_paths)}",
            "",
            "[git_auth]",
            f"token_env = {_quote(config.git_auth.token_env)}",
            f"username = {_quote(config.git_auth.username)}",
            f"use_for_local_push = {str(config.git_auth.use_for_local_push).lower()}",
            f"use_for_remote_fetch = {str(config.git_auth.use_for_remote_fetch).lower()}",
            "",
            "[runner]",
            f"mode = {_quote(config.runner.mode)}",
        ]
    )
    if config.git_auth.fallback_token_env:
        lines.insert(lines.index("[runner]"), f"fallback_token_env = {_quote(config.git_auth.fallback_token_env)}")
    lines.append("")
    lines.append("[runner.local]")
    if config.runner.local.activation_command:
        lines.append(f"activation_command = {_quote(config.runner.local.activation_command)}")
    if config.runner.local.env:
        lines.append("")
        lines.append("[runner.local.env]")
        lines.extend(_render_str_map(config.runner.local.env))
    lines.append("")
    lines.append("[runner.ssh]")
    if config.runner.ssh.host:
        lines.append(f"host = {_quote(config.runner.ssh.host)}")
    if config.runner.ssh.user:
        lines.append(f"user = {_quote(config.runner.ssh.user)}")
    lines.append(f"port = {config.runner.ssh.port}")
    if config.runner.ssh.key_path:
        lines.append(f"key_path = {_render_path_value(config.runner.ssh.key_path, config.repo_root)}")
    if config.runner.ssh.host_alias:
        lines.append(f"host_alias = {_quote(config.runner.ssh.host_alias)}")
    lines.extend(
        [
            f"repo_path = {_quote(config.runner.ssh.repo_path)}",
            f"git_remote = {_quote(config.runner.ssh.git_remote)}",
            f"runtime_root = {_quote(config.runner.ssh.runtime_root)}",
        ]
    )
    if config.runner.ssh.activation_command:
        lines.append(f"activation_command = {_quote(config.runner.ssh.activation_command)}")
    if config.runner.ssh.env:
        lines.append("")
        lines.append("[runner.ssh.env]")
        lines.extend(_render_str_map(config.runner.ssh.env))
    for stage_name in ("scout", "confirm", "promoted"):
        stage = getattr(config.runner, stage_name)
        lines.extend(
            [
                "",
                f"[runner.{stage_name}]",
                f"enabled = {str(stage.enabled).lower()}",
                f"commands = {_render_list(stage.commands)}",
                f"timeout_seconds = {stage.timeout_seconds}",
            ]
        )
        if stage.env:
            lines.append("")
            lines.append(f"[runner.{stage_name}.env]")
            lines.extend(_render_str_map(stage.env))
    for metric in config.metrics:
        lines.extend(["", *_render_metric(metric)])
    lines.extend(
        [
            "",
            "[objective]",
            f"primary_metric = {_quote(config.objective.primary_metric)}",
            f"direction = {_quote(config.objective.direction)}",
            f"minimum_improvement_delta = {config.objective.minimum_improvement_delta}",
            "",
            "[search]",
            f"oracle = {_quote(config.search.oracle)}",
            f"max_history = {config.search.max_history}",
            f"parent_suggestion_count = {config.search.parent_suggestion_count}",
            f"family_suggestion_count = {config.search.family_suggestion_count}",
            f"knob_region_count = {config.search.knob_region_count}",
            f"risk_penalty = {config.search.risk_penalty}",
            f"family_bonus = {config.search.family_bonus}",
            f"lineage_bonus = {config.search.lineage_bonus}",
            f"exploration_weight = {config.search.exploration_weight}",
            f"observation_noise = {config.search.observation_noise}",
        ]
    )
    if config.objective.threshold is not None:
        objective_index = lines.index("[objective]")
        lines.insert(objective_index + 3, f"threshold = {config.objective.threshold}")
    if config.objective.cost_penalty_metric:
        objective_index = lines.index("[objective]")
        lines.insert(objective_index + 4, f"cost_penalty_metric = {_quote(config.objective.cost_penalty_metric)}")
        lines.insert(objective_index + 5, f"cost_penalty_weight = {config.objective.cost_penalty_weight}")
    if config.search.seed is not None:
        search_index = lines.index("[search]")
        lines.insert(search_index + 2, f"seed = {config.search.seed}")
    return "\n".join(lines) + "\n"


def render_boa_md(plan: ReviewedInitPlan) -> str:
    surfaces = ", ".join(plan.analysis.optimization_surfaces) if plan.analysis.optimization_surfaces else "repo hotspots"
    caveats = "\n".join(f"- {item}" for item in plan.analysis.caveats) or "- Confirm commands and metric extraction before long runs."
    aggressiveness = _normalize_aggressiveness(plan.selection.agent_aggressiveness)
    aggressiveness_rules = "\n".join(f"- {item}" for item in _aggressiveness_rules(aggressiveness))
    return render_prompt_template(
        "init",
        "boa_md_template.md",
        base_boa_md=plan.boa_md.strip(),
        aggressiveness=aggressiveness,
        aggressiveness_summary=_aggressiveness_summary(aggressiveness),
        aggressiveness_rules=aggressiveness_rules,
        train_command=plan.train_command,
        eval_command=plan.eval_command,
        primary_metric_name=plan.primary_metric_name,
        metric_direction=plan.metric_direction,
        editable_paths=", ".join(plan.editable_files),
        protected_paths=", ".join(plan.protected_files),
        optimization_surfaces=surfaces,
        caveats=caveats,
    )


def write_contract_files(plan: ReviewedInitPlan) -> WriteResult:
    result = WriteResult()
    config_path = plan.repo_root / "boa.config"
    boa_md_path = plan.repo_root / "boa.md"
    runtime_root = plan.repo_root / ".boa"
    if plan.selection.existing_action == "review":
        result.skipped_paths.extend([config_path, boa_md_path])
        return result
    config_existed = config_path.exists()
    boa_md_existed = boa_md_path.exists()
    runtime_existed = runtime_root.exists()
    runtime_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_config_text(plan), encoding="utf-8")
    boa_md_path.write_text(render_boa_md(plan), encoding="utf-8")
    if config_existed:
        result.updated_paths.append(config_path)
    else:
        result.created_paths.append(config_path)
    if boa_md_existed:
        result.updated_paths.append(boa_md_path)
    else:
        result.created_paths.append(boa_md_path)
    if runtime_existed:
        result.updated_paths.append(runtime_root)
    else:
        result.created_paths.append(runtime_root)
    return result


def validate_written_setup(plan: ReviewedInitPlan) -> ValidationReport:
    details: list[str] = []
    config = load_config(plan.repo_root)
    details.append("Config parses successfully.")
    build_trial_runner(config, git_auth=None)
    details.append(f"Runner mode '{config.runner.mode}' validated.")
    if config.agent.runtime == "cli":
        binary = shlex.split(config.agent.command or "")[:1]
        if not binary or shutil.which(binary[0]) is None:
            raise RuntimeError(f"Configured agent command is not callable: {config.agent.command}")
    else:
        if importlib.util.find_spec("deepagents") is None:
            raise RuntimeError("deepagents is not installed")
    details.append(f"Agent '{config.agent.preset}' validated.")
    return ValidationReport(passed=True, details=details)


@dataclass
class InitServices:
    detect_repo: Callable[[Path], DetectedRepo] = detect_repo
    default_selection_for_repo: Callable[[DetectedRepo], InitSetupSelection] = default_selection_for_repo
    run_preflight: Callable[[DetectedRepo, InitSetupSelection], list[PreflightCheck]] = run_preflight
    analyze_repo: Callable[[DetectedRepo, InitSetupSelection], RepoAnalysisProposal] = analyze_repo
    default_repo_analysis: Callable[[Path], RepoAnalysisProposal] = default_repo_analysis
    merge_reviewed_plan: Callable[[InitSetupSelection, RepoAnalysisProposal], ReviewedInitPlan] = merge_reviewed_plan
    write_contract_files: Callable[[ReviewedInitPlan], WriteResult] = write_contract_files
    validate_written_setup: Callable[[ReviewedInitPlan], ValidationReport] = validate_written_setup
