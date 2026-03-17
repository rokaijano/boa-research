from __future__ import annotations

from .base import BaseResearchAgent, ResearchAgentError


def build_agent(
    config,
    *,
    repo_root,
    run_preflight,
    read_recent_trials,
) -> BaseResearchAgent:
    runtime = str(config.agent.runtime).strip().lower()
    if runtime == "deepagents":
        from .deepagents import DeepAgentsResearchAgent

        return DeepAgentsResearchAgent(
            repo_root=repo_root,
            agent_config=config.agent,
            run_preflight=run_preflight,
            read_recent_trials=read_recent_trials,
        )
    if runtime == "cli":
        from .cli import CliResearchAgent

        return CliResearchAgent(repo_root=repo_root, agent_config=config.agent)
    raise ValueError(f"Unsupported agent.runtime: {config.agent.runtime}")


__all__ = [
    "BaseResearchAgent",
    "ResearchAgentError",
    "build_agent",
]
