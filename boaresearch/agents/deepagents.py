from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Optional

from ..prompt_builder import build_agent_system_prompt, build_agent_task
from ..schema import CandidateMetadata
from .base import BaseResearchAgent, ResearchAgentError
from .tools import AgentToolHarness, CandidateSubmissionRecorder, build_agent_tools


def _import_attr(module_names: list[str], attr: str) -> Any:
    last_error: Optional[Exception] = None
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            last_error = exc
            continue
        if hasattr(module, attr):
            return getattr(module, attr)
    raise ResearchAgentError(f"Unable to import {attr} from {module_names}: {last_error}")


def _call_with_supported_kwargs(callable_obj, /, *args, **kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(*args, **kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return callable_obj(*args, **filtered)


class DeepAgentsResearchAgent(BaseResearchAgent):
    def __init__(
        self,
        *,
        repo_root: Path,
        agent_config,
        run_preflight,
        read_recent_trials,
    ) -> None:
        self.repo_root = repo_root
        self.agent_config = agent_config
        self._run_preflight = run_preflight
        self._read_recent_trials = read_recent_trials

    def _build_model(self):
        init_chat_model = _import_attr(
            ["langchain.chat_models", "langchain.chat_models.base"],
            "init_chat_model",
        )
        model_name = str(self.agent_config.model).strip()
        if model_name and not model_name.startswith("openai:") and not model_name.startswith("ollama:"):
            model_name = f"{self.agent_config.backend}:{model_name}"
        kwargs: dict[str, Any] = {
            "base_url": str(self.agent_config.base_url).strip(),
            "max_tokens": int(self.agent_config.max_output_tokens),
            "max_output_tokens": int(self.agent_config.max_output_tokens),
        }
        if self.agent_config.reasoning_effort:
            kwargs["reasoning_effort"] = str(self.agent_config.reasoning_effort)
        kwargs.update(dict(self.agent_config.provider_options))
        if self.agent_config.backend == "openai":
            api_key_env = self.agent_config.api_key_env or "OPENAI_API_KEY"
            api_key = str(os.environ.get(api_key_env, "")).strip()
            if not api_key:
                raise ResearchAgentError(f"Missing API key environment variable: {api_key_env}")
            kwargs["api_key"] = api_key
        return _call_with_supported_kwargs(init_chat_model, model_name, **kwargs)

    def _build_backend(self, *, worktree_path: Path, memories_path: Path):
        filesystem_backend_cls = _import_attr(
            ["deepagents.backends", "deepagents"],
            "FilesystemBackend",
        )
        composite_backend_cls = _import_attr(
            ["deepagents.backends", "deepagents"],
            "CompositeBackend",
        )
        default_backend = _call_with_supported_kwargs(
            filesystem_backend_cls,
            root_dir=str(worktree_path),
            virtual_mode=True,
        )
        memories_backend = _call_with_supported_kwargs(
            filesystem_backend_cls,
            root_dir=str(memories_path),
            virtual_mode=True,
        )
        return _call_with_supported_kwargs(
            composite_backend_cls,
            default=default_backend,
            default_backend=default_backend,
            routes={"/memories/": memories_backend},
        )

    def _create_agent(self, *, model, backend, tools, system_prompt: str, max_agent_steps: int):
        create_deep_agent = _import_attr(["deepagents"], "create_deep_agent")
        return _call_with_supported_kwargs(
            create_deep_agent,
            model=model,
            tools=tools,
            backend=backend,
            system_prompt=system_prompt,
            instructions=system_prompt,
            max_iterations=max_agent_steps,
            max_steps=max_agent_steps,
            max_agent_steps=max_agent_steps,
        )

    @staticmethod
    def _invoke_agent(agent: Any, task: str) -> Any:
        payload = {"messages": [{"role": "user", "content": task}]}
        if hasattr(agent, "invoke"):
            return agent.invoke(payload)
        if callable(agent):
            return agent(payload)
        raise ResearchAgentError("DeepAgents runtime returned a non-invokable agent object")

    def prepare_candidate(self, context) -> CandidateMetadata:
        recorder = CandidateSubmissionRecorder()
        harness = AgentToolHarness(
            run_preflight=self._run_preflight,
            read_recent_trials=self._read_recent_trials,
            submission=recorder,
        )
        tools = build_agent_tools(harness)
        memories_path = self.repo_root / ".boa" / "runtime" / "deepagents" / context.run_tag / "memories"
        memories_path.mkdir(parents=True, exist_ok=True)
        model = self._build_model()
        backend = self._build_backend(worktree_path=context.worktree_path, memories_path=memories_path)
        system_prompt = build_agent_system_prompt(repo_root=self.repo_root, context=context)
        task = build_agent_task(context)
        agent = self._create_agent(
            model=model,
            backend=backend,
            tools=tools,
            system_prompt=system_prompt,
            max_agent_steps=context.max_agent_steps,
        )
        self._invoke_agent(agent, task)
        if recorder.candidate is None:
            raise ResearchAgentError("DeepAgents must call submit_candidate() before returning control")
        return recorder.candidate
