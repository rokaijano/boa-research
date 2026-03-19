from __future__ import annotations

import json
import os
import html
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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


_MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_LOCAL_OLLAMA_BASE_URLS = {"http://127.0.0.1:11434", "http://localhost:11434"}
_COPILOT_SUPPORTED_MODELS_URL = "https://docs.github.com/en/copilot/reference/ai-models/supported-models"
_COPILOT_DOC_MODEL_DISPLAY_PATTERN = re.compile(
    r"\b(?:GPT-[0-9.]+(?:-Codex(?:-Mini|-Max)?)?(?: mini)?|Claude (?:Haiku|Opus|Sonnet) [0-9.]+|Gemini [0-9.]+(?:\.[0-9]+)? (?:Pro|Flash)|Grok Code Fast 1|Raptor mini|Goldeneye)\b"
)
_COPILOT_DOC_ROW_PATTERN = re.compile(
    r'<th[^>]*scope="row"[^>]*>\s*([^<]+?)\s*</th>\s*<td[^>]*>\s*(OpenAI|Anthropic|Google|xAI|Fine-tuned [^<]+?)\s*</td>',
    re.IGNORECASE | re.DOTALL,
)
_CODEX_MODELS_URL = "https://developers.openai.com/codex/models"
_CODEX_DOC_MODEL_PATTERN = re.compile(r"codex\s+-m\s+([A-Za-z0-9][A-Za-z0-9._:-]*)")
_CLAUDE_MODELS_URL = "https://platform.claude.com/docs/en/about-claude/models/overview"
_CLAUDE_DOC_MODEL_PATTERN = re.compile(r"\b(claude-(?:opus|sonnet|haiku)-[A-Za-z0-9.-]+)\b")
_CLAUDE_DOC_BEDROCK_SUFFIX_PATTERN = re.compile(r"-v\d+$")
_TRAILING_PARENTHETICAL_PATTERN = re.compile(r"\s*\([^)]*\)")
_DEFAULT_MODEL_AVAILABILITY_TIMEOUT_SECONDS = 20
_COPILOT_MODEL_AVAILABILITY_TIMEOUT_SECONDS = 30
_COPILOT_MODEL_AVAILABILITY_RETRY_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class ModelDiscoveryResult:
    models: list[tuple[str, str]]
    source: str | None = None
    warning: str | None = None
    supports_listing: bool = True


@dataclass(frozen=True)
class ModelAvailabilityResult:
    status: str
    message: str | None = None
    source: str | None = None


def _dedupe_choices(choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, value in choices:
        if value in seen:
            continue
        seen.add(value)
        deduped.append((label, value))
    return deduped


def _dedupe_model_names(model_names: list[str]) -> list[str]:
    return list(dict.fromkeys(model_name for model_name in model_names if model_name))


def _normalize_command(command: str) -> str:
    normalized = str(command or "").strip()
    return normalized or "codex"


def _command_exists(command: str) -> bool:
    executable = _normalize_command(command)
    return shutil.which(executable) is not None or Path(executable).exists()


def _resolve_command_path(command: str) -> str:
    executable = _normalize_command(command)
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    return executable


def _build_command_argv(command: str, args: list[str]) -> list[str]:
    executable = _resolve_command_path(command)
    if os.name == "nt" and executable.lower().endswith(".ps1"):
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            executable,
            *args,
        ]
    return [executable, *args]


def _parse_model_tokens(text: str) -> list[str]:
    models: list[str] = []
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip().lstrip("-*•").strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("available model"):
            continue
        matches = _MODEL_NAME_PATTERN.findall(stripped)
        if not matches:
            continue
        candidate = matches[0]
        if candidate.lower() in {"model", "models", "name"}:
            if len(matches) < 2:
                continue
            candidate = matches[1]
        models.append(candidate)
    return _dedupe_model_names(models)


def _models_endpoint(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/models"):
        return normalized
    return f"{normalized}/models"


def _ollama_tags_endpoint(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/api/tags"):
        return normalized
    return f"{normalized}/api/tags"


def _read_json_response(*, url: str, headers: dict[str, str] | None = None, timeout_seconds: int = 5) -> Any:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _read_text_response(*, url: str, headers: dict[str, str] | None = None, timeout_seconds: int = 5) -> str:
    request = urllib.request.Request(url, headers=dict(headers or {}))
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def _read_github_docs_article_body(*, url: str, timeout_seconds: int = 5) -> str:
    parsed = urllib.parse.urlparse(url)
    pathname = parsed.path or "/"
    api_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/api/article", "", urllib.parse.urlencode({"pathname": pathname}), ""))
    payload = _read_json_response(
        url=api_url,
        headers={
            "User-Agent": "boa-research-copilot-model-helper/1.0",
            "Accept": "application/json",
        },
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(payload, dict):
        return ""
    body = payload.get("body")
    if not isinstance(body, str):
        return ""
    return body


def _slice_text_between_markers(text: str, *, start_marker: str, end_marker: str, search_start: int = 0) -> str:
    start_index = text.find(start_marker, max(0, search_start))
    if start_index < 0:
        return text
    end_index = text.find(end_marker, start_index + len(start_marker))
    if end_index < 0:
        return text[start_index:]
    return text[start_index:end_index]


def _read_api_model_ids(*, base_url: str, headers: dict[str, str], timeout_seconds: int = 5) -> list[str]:
    payload = _read_json_response(url=_models_endpoint(base_url), headers=headers, timeout_seconds=timeout_seconds)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    model_names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if model_id:
            model_names.append(model_id)
    return _dedupe_model_names(model_names)


def _run_model_discovery_command(*, command: str, args: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str] | None:
    if not _command_exists(command):
        return None
    try:
        return subprocess.run(
            _build_command_argv(command, args),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def probe_codex_model_discovery(*, command: str = "codex", timeout_seconds: int = 5) -> ModelDiscoveryResult:
    proc = _run_model_discovery_command(command=command, args=["prompt", "--models"], timeout_seconds=timeout_seconds)
    if proc is None:
        return ModelDiscoveryResult(
            models=[],
            source="codex prompt --models",
            warning="BOA could not launch Codex to inspect models. You can still enter a model manually.",
        )
    if proc.returncode == 0:
        models = _parse_model_tokens(proc.stdout)
        if models:
            return ModelDiscoveryResult(models=[(model_name, model_name) for model_name in models], source="codex prompt --models")
        return ModelDiscoveryResult(
            models=[],
            source="codex prompt --models",
            warning="Codex returned no models. You can still enter a model manually.",
        )
    stderr = str(proc.stderr or "").strip()
    if "unexpected argument '--models'" in stderr:
        warning = "The installed Codex CLI does not support `codex prompt --models`. You can still enter a model manually."
    else:
        warning = "BOA could not load models from Codex. You can still enter a model manually."
    return ModelDiscoveryResult(models=[], source="codex prompt --models", warning=warning)


def probe_codex_model_availability(*, command: str = "codex", model: str, timeout_seconds: int = 20) -> ModelAvailabilityResult:
    selected_model = str(model or "").strip()
    if not selected_model:
        return ModelAvailabilityResult(status="available")
    proc = _run_model_discovery_command(
        command=command,
        args=[
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
            "--ephemeral",
            "-m",
            selected_model,
            "Reply with OK only.",
        ],
        timeout_seconds=timeout_seconds,
    )
    if proc is None:
        return ModelAvailabilityResult(
            status="unknown",
            message="BOA could not launch Codex to verify the selected model.",
        )
    combined_output = "\n".join(part for part in [str(proc.stdout or "").strip(), str(proc.stderr or "").strip()] if part).strip()
    if proc.returncode == 0:
        return ModelAvailabilityResult(status="available", source="codex exec probe")
    lowered_output = combined_output.lower()
    if (
        "model is not supported" in lowered_output
        or "model metadata" in lowered_output and "not found" in lowered_output
        or "not supported when using codex" in lowered_output
    ):
        return ModelAvailabilityResult(
            status="unavailable",
            source="codex exec probe",
            message=f"`{selected_model}` is not available via Codex. Choose another model.",
        )
    return ModelAvailabilityResult(
        status="unknown",
        source="codex exec probe",
        message="BOA could not conclusively verify Codex model availability non-interactively. Proceeding with the selected model.",
    )


def probe_copilot_model_availability(*, command: str = "copilot", model: str, timeout_seconds: int = 20) -> ModelAvailabilityResult:
    selected_model = str(model or "").strip()
    if not selected_model:
        return ModelAvailabilityResult(status="available")
    proc = _run_model_discovery_command(
        command=command,
        args=[
            "-p",
            "Reply with OK only.",
            "--model",
            selected_model,
            "--allow-all-tools",
            "--output-format",
            "json",
        ],
        timeout_seconds=timeout_seconds,
    )
    if proc is None:
        return ModelAvailabilityResult(
            status="unknown",
            message="BOA could not launch Copilot to verify the selected model.",
        )
    combined_output = "\n".join(part for part in [str(proc.stdout or "").strip(), str(proc.stderr or "").strip()] if part).strip()
    if proc.returncode == 0:
        return ModelAvailabilityResult(status="available", source="copilot prompt probe")
    lowered_output = combined_output.lower()
    if 'from --model flag is not available' in lowered_output:
        return ModelAvailabilityResult(
            status="unavailable",
            source="copilot prompt probe",
            message=f"`{selected_model}` is not available via Copilot. Choose another model.",
        )
    return ModelAvailabilityResult(
        status="unknown",
        source="copilot prompt probe",
        message="BOA could not conclusively verify Copilot model availability non-interactively. Proceeding with the selected model.",
    )


def discover_codex_models(*, command: str = "codex", timeout_seconds: int = 5) -> list[str]:
    result = probe_codex_model_discovery(command=command, timeout_seconds=timeout_seconds)
    return [value for _, value in result.models]


def discover_codex_doc_models(*, url: str = _CODEX_MODELS_URL, timeout_seconds: int = 5) -> list[str]:
    try:
        page_text = _read_text_response(
            url=url,
            headers={
                "User-Agent": "boa-research-codex-model-helper/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout_seconds=timeout_seconds,
        )
    except (OSError, urllib.error.URLError):
        return []
    decoded = html.unescape(str(page_text or ""))
    models = _CODEX_DOC_MODEL_PATTERN.findall(decoded)
    return _dedupe_model_names(models)


def _normalize_copilot_doc_model_name(display_name: str) -> str:
    normalized = _TRAILING_PARENTHETICAL_PATTERN.sub("", str(display_name or "")).strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("claude "):
        parts = normalized.split()
        if len(parts) >= 3:
            return f"claude-{parts[1]}-{parts[2]}"
    return normalized.replace(" ", "-")


def discover_copilot_doc_models(*, url: str = _COPILOT_SUPPORTED_MODELS_URL, timeout_seconds: int = 5) -> list[str]:
    markdown_body = ""
    try:
        markdown_body = _read_github_docs_article_body(url=url, timeout_seconds=timeout_seconds)
    except (OSError, ValueError, urllib.error.URLError):
        markdown_body = ""
    markdown_section = _slice_text_between_markers(
        markdown_body,
        start_marker="Supported AI models in Copilot",
        end_marker="Model retirement history",
    )
    markdown_matches = _COPILOT_DOC_MODEL_DISPLAY_PATTERN.findall(markdown_section)
    models = [_normalize_copilot_doc_model_name(display_name) for display_name in markdown_matches]
    models = _dedupe_model_names(models)
    if models:
        return models
    try:
        page_text = _read_text_response(
            url=url,
            headers={
                "User-Agent": "boa-research-copilot-model-helper/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout_seconds=timeout_seconds,
        )
    except (OSError, urllib.error.URLError):
        return []
    decoded_html = html.unescape(str(page_text or ""))
    article_index = decoded_html.find('data-container="article"')
    supported_section = _slice_text_between_markers(
        decoded_html,
        start_marker="Supported AI models in Copilot",
        end_marker="Model retirement history",
        search_start=article_index if article_index >= 0 else 0,
    )
    html_matches = [model_name for model_name, _provider in _COPILOT_DOC_ROW_PATTERN.findall(supported_section)]
    models = [_normalize_copilot_doc_model_name(display_name) for display_name in html_matches]
    return _dedupe_model_names(models)


def discover_anthropic_models(
    *,
    api_key_env: str = "ANTHROPIC_API_KEY",
    base_url: str = "https://api.anthropic.com/v1",
    timeout_seconds: int = 5,
) -> list[str]:
    api_key = str(os.environ.get(api_key_env, "")).strip()
    if not api_key:
        return []
    try:
        return _read_api_model_ids(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout_seconds=timeout_seconds,
        )
    except (OSError, ValueError, urllib.error.URLError):
        return []


def discover_claude_doc_models(*, url: str = _CLAUDE_MODELS_URL, timeout_seconds: int = 5) -> list[str]:
    try:
        page_text = _read_text_response(
            url=url,
            headers={
                "User-Agent": "boa-research-claude-model-helper/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout_seconds=timeout_seconds,
        )
    except (OSError, urllib.error.URLError):
        return []
    decoded = html.unescape(str(page_text or ""))
    extracted = _CLAUDE_DOC_MODEL_PATTERN.findall(decoded)
    models: list[str] = []
    for model_name in extracted:
        normalized = _CLAUDE_DOC_BEDROCK_SUFFIX_PATTERN.sub("", model_name)
        models.append(normalized)
    return _dedupe_model_names(models)


def discover_openai_models(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    timeout_seconds: int = 5,
) -> list[str]:
    api_key = str(os.environ.get(api_key_env, "")).strip()
    if not api_key:
        return []
    try:
        return _read_api_model_ids(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_seconds=timeout_seconds,
        )
    except (OSError, ValueError, urllib.error.URLError):
        return []


def discover_ollama_models(*, base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 3) -> list[str]:
    normalized_base_url = str(base_url or "http://127.0.0.1:11434").strip().rstrip("/")
    try:
        payload = _read_json_response(url=_ollama_tags_endpoint(normalized_base_url), timeout_seconds=timeout_seconds)
    except (OSError, ValueError, urllib.error.URLError):
        payload = None
    if isinstance(payload, dict):
        payload_models = payload.get("models")
        if isinstance(payload_models, list):
            model_names: list[str] = []
            for item in payload_models:
                if not isinstance(item, dict):
                    continue
                model_name = str(item.get("model") or item.get("name") or "").strip()
                if model_name:
                    model_names.append(model_name)
            if model_names:
                return _dedupe_model_names(model_names)
    if normalized_base_url not in _LOCAL_OLLAMA_BASE_URLS or shutil.which("ollama") is None:
        return []
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    models: list[str] = []
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name "):
            continue
        model_name = stripped.split()[0].strip()
        if model_name and model_name.upper() != "NAME":
            models.append(model_name)
    return _dedupe_model_names(models)


def discover_available_models_for_agent(
    *,
    preset: str,
    backend: str | None = None,
    command: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int = 5,
) -> ModelDiscoveryResult:
    preset_name = str(preset or "").strip().lower()
    if preset_name == "codex":
        cli_discovery = probe_codex_model_discovery(command=command or "codex", timeout_seconds=timeout_seconds)
        if cli_discovery.models:
            return cli_discovery
        doc_models = discover_codex_doc_models(timeout_seconds=timeout_seconds)
        if doc_models:
            warning = "Loaded documented Codex models from the public docs page; actual Codex availability may still depend on your account or local configuration."
            if cli_discovery.warning and "does not support `codex prompt --models`" in cli_discovery.warning:
                warning = "The installed Codex CLI does not support `codex prompt --models`, so BOA loaded documented Codex models from the public docs page; actual availability may still depend on your account or local configuration."
            return ModelDiscoveryResult(
                models=[(model_name, model_name) for model_name in doc_models],
                source="OpenAI Codex models docs",
                warning=warning,
            )
        return cli_discovery
    if preset_name == "claude_code":
        key_env = api_key_env or "ANTHROPIC_API_KEY"
        models = discover_anthropic_models(api_key_env=key_env, timeout_seconds=timeout_seconds)
        if models:
            return ModelDiscoveryResult(models=[(model_name, model_name) for model_name in models], source="Anthropic /v1/models")
        doc_models = discover_claude_doc_models(timeout_seconds=timeout_seconds)
        if doc_models:
            return ModelDiscoveryResult(
                models=[(model_name, model_name) for model_name in doc_models],
                source="Anthropic docs models overview",
                warning="Loaded documented Claude models from the public docs page; actual CLI availability may still depend on your account or org policy.",
            )
        if not str(os.environ.get(key_env, "")).strip():
            warning = f"Set {key_env} to load Claude models dynamically, or enter a model manually."
        else:
            warning = "BOA could not load models from Anthropic /v1/models or the public docs page. You can still enter a model manually."
        return ModelDiscoveryResult(models=[], source="Anthropic /v1/models", warning=warning)
    if preset_name == "copilot":
        doc_models = discover_copilot_doc_models(timeout_seconds=timeout_seconds)
        if doc_models:
            return ModelDiscoveryResult(
                models=[(model_name, model_name) for model_name in doc_models],
                source="GitHub Copilot supported models docs",
                warning="Loaded documented Copilot models from GitHub Docs; actual availability depends on your plan, client, and organization policy.",
            )
        return ModelDiscoveryResult(
            models=[],
            warning="Copilot CLI does not expose a non-interactive model list, and BOA could not load the GitHub Docs fallback. Enter a model manually here.",
            supports_listing=False,
        )
    if preset_name == "deepagents":
        backend_name = str(backend or "ollama").strip().lower()
        if backend_name == "ollama":
            resolved_base_url = base_url or "http://127.0.0.1:11434"
            models = discover_ollama_models(base_url=resolved_base_url, timeout_seconds=timeout_seconds)
            if models:
                return ModelDiscoveryResult(models=[(model_name, model_name) for model_name in models], source=f"{resolved_base_url}/api/tags")
            return ModelDiscoveryResult(
                models=[],
                source=f"{resolved_base_url}/api/tags",
                warning="BOA could not load models from Ollama. You can still enter a model manually.",
            )
        if backend_name == "openai":
            resolved_base_url = base_url or "https://api.openai.com/v1"
            key_env = api_key_env or "OPENAI_API_KEY"
            models = discover_openai_models(api_key_env=key_env, base_url=resolved_base_url, timeout_seconds=timeout_seconds)
            if models:
                return ModelDiscoveryResult(models=[(model_name, model_name) for model_name in models], source=f"{resolved_base_url}/models")
            if not str(os.environ.get(key_env, "")).strip():
                warning = f"Set {key_env} to load OpenAI-compatible models dynamically, or enter a model manually."
            else:
                warning = "BOA could not load models from the OpenAI-compatible /models endpoint. You can still enter a model manually."
            return ModelDiscoveryResult(models=[], source=f"{resolved_base_url}/models", warning=warning)
    return ModelDiscoveryResult(models=[])


def _availability_from_model_list(*, model: str, models: list[str], source: str) -> ModelAvailabilityResult:
    if model in models:
        return ModelAvailabilityResult(status="available", source=source)
    return ModelAvailabilityResult(
        status="unavailable",
        source=source,
        message=f"`{model}` is not available via {source}. Choose another model.",
    )


def check_model_availability_for_agent(
    *,
    preset: str,
    model: str,
    backend: str | None = None,
    command: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int = 5,
) -> ModelAvailabilityResult:
    if not str(model or "").strip():
        return ModelAvailabilityResult(status="available")

    preset_name = str(preset or "").strip().lower()
    selected_model = str(model).strip()
    availability_timeout_seconds = max(timeout_seconds, _DEFAULT_MODEL_AVAILABILITY_TIMEOUT_SECONDS)

    if preset_name == "codex":
        probe_result = probe_codex_model_availability(
            command=command or "codex",
            model=selected_model,
            timeout_seconds=availability_timeout_seconds,
        )
        if probe_result.status != "unknown":
            return probe_result
        cli_models = discover_codex_models(command=command or "codex", timeout_seconds=timeout_seconds)
        if cli_models:
            return _availability_from_model_list(model=selected_model, models=cli_models, source="codex prompt --models")
        key_env = api_key_env or "OPENAI_API_KEY"
        openai_models = discover_openai_models(api_key_env=key_env, timeout_seconds=timeout_seconds)
        if openai_models:
            return _availability_from_model_list(model=selected_model, models=openai_models, source="OpenAI /models")
        doc_models = discover_codex_doc_models(timeout_seconds=timeout_seconds)
        if doc_models:
            return _availability_from_model_list(model=selected_model, models=doc_models, source="OpenAI Codex models docs")
        return ModelAvailabilityResult(
            status="unknown",
            message="BOA could not verify Codex model availability non-interactively. Proceeding with the selected model.",
        )

    if preset_name == "claude_code":
        key_env = api_key_env or "ANTHROPIC_API_KEY"
        anthropic_models = discover_anthropic_models(api_key_env=key_env, timeout_seconds=timeout_seconds)
        if anthropic_models:
            return _availability_from_model_list(model=selected_model, models=anthropic_models, source="Anthropic /v1/models")
        return ModelAvailabilityResult(
            status="unknown",
            message="BOA could not verify Claude model availability for this account non-interactively. Proceeding with the selected model.",
        )

    if preset_name == "deepagents":
        backend_name = str(backend or "ollama").strip().lower()
        if backend_name == "ollama":
            resolved_base_url = base_url or "http://127.0.0.1:11434"
            ollama_models = discover_ollama_models(base_url=resolved_base_url, timeout_seconds=timeout_seconds)
            if ollama_models:
                return _availability_from_model_list(model=selected_model, models=ollama_models, source=f"{resolved_base_url}/api/tags")
            return ModelAvailabilityResult(
                status="unknown",
                message="BOA could not verify Ollama model availability right now. Proceeding with the selected model.",
            )
        if backend_name == "openai":
            resolved_base_url = base_url or "https://api.openai.com/v1"
            key_env = api_key_env or "OPENAI_API_KEY"
            openai_models = discover_openai_models(
                api_key_env=key_env,
                base_url=resolved_base_url,
                timeout_seconds=timeout_seconds,
            )
            if openai_models:
                return _availability_from_model_list(model=selected_model, models=openai_models, source=f"{resolved_base_url}/models")
            return ModelAvailabilityResult(
                status="unknown",
                message="BOA could not verify the OpenAI-compatible model availability right now. Proceeding with the selected model.",
            )

    if preset_name == "copilot":
        copilot_timeout_seconds = max(timeout_seconds, _COPILOT_MODEL_AVAILABILITY_TIMEOUT_SECONDS)
        probe_result = probe_copilot_model_availability(
            command=command or "copilot",
            model=selected_model,
            timeout_seconds=copilot_timeout_seconds,
        )
        if probe_result.status == "unknown":
            probe_result = probe_copilot_model_availability(
                command=command or "copilot",
                model=selected_model,
                timeout_seconds=max(copilot_timeout_seconds, _COPILOT_MODEL_AVAILABILITY_RETRY_TIMEOUT_SECONDS),
            )
        if probe_result.status != "unknown":
            return probe_result
        return ModelAvailabilityResult(
            status="unknown",
            message="BOA cannot verify Copilot model availability non-interactively. Proceeding with the selected model.",
        )

    return ModelAvailabilityResult(status="unknown")


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
