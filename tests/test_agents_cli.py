from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from boaresearch.agents.cli import CliResearchAgent
from boaresearch.agents.deepagents import DeepAgentsResearchAgent
from boaresearch.prompt_builder import CANDIDATE_SCHEMA, PLAN_SCHEMA, build_execution_system_prompt, build_planning_system_prompt
from boaresearch.prompt_builder import build_execution_task, build_planning_task
from boaresearch.runtime.observer import RunEvent
from boaresearch.schema import (
    AgentConfig,
    AgentExecutionContext,
    AgentPlanningContext,
    BoaConfig,
    CandidatePlan,
    MetricConfig,
    ObjectiveConfig,
    RunConfig,
    RunnerConfig,
    RunnerStageConfig,
    SearchConfig,
)
from boaresearch.search import SearchToolContext, write_search_tool_context


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


def _write_tool_context_file(path: Path, repo_root: Path, phase: str) -> None:
    write_search_tool_context(
        path,
        SearchToolContext(
            repo_root=repo_root,
            config_path=repo_root / "boa.config",
            run_tag="demo",
            trial_id="demo-0001",
            accepted_branch="boa/demo/accepted",
            phase=phase,
            trace_path=repo_root / "trace.jsonl",
        ),
    )


def _make_repo() -> Path:
    repo = Path(tempfile.mkdtemp())
    (repo / "worktree").mkdir()
    (repo / "boa.md").write_text("# BOA\n", encoding="utf-8")
    (repo / "boa.config").write_text("schema_version = 3\n", encoding="utf-8")
    return repo


def _planning_context(repo: Path) -> AgentPlanningContext:
    prompt_dir = repo / "prompts" / "planning"
    tool_context_path = prompt_dir / "tool_context.json"
    _write_tool_context_file(tool_context_path, repo, "planning")
    return AgentPlanningContext(
        repo_root=repo,
        worktree_path=repo / "worktree",
        trial_id="demo-0001",
        run_tag="demo",
        accepted_branch="boa/demo/accepted",
        boa_md_path=repo / "boa.md",
        extra_context_files=[],
        allowed_paths=["src"],
        protected_paths=[".boa"],
        recent_trials=[],
        objective_summary="Objective",
        max_agent_steps=10,
        prompt_bundle_dir=prompt_dir,
        tool_context_path=tool_context_path,
        plan_output_path=repo / "plan.json",
    )


def _execution_context(repo: Path) -> AgentExecutionContext:
    prompt_dir = repo / "prompts" / "execution"
    tool_context_path = prompt_dir / "tool_context.json"
    _write_tool_context_file(tool_context_path, repo, "execution")
    return AgentExecutionContext(
        repo_root=repo,
        worktree_path=repo / "worktree",
        trial_id="demo-0001",
        run_tag="demo",
        accepted_branch="boa/demo/accepted",
        trial_branch="boa/demo/trial/demo-0001",
        parent_branch="boa/demo/accepted",
        parent_trial_id=None,
        boa_md_path=repo / "boa.md",
        extra_context_files=[],
        allowed_paths=["src"],
        protected_paths=[".boa"],
        recent_trials=[],
        objective_summary="Objective",
        preflight_commands=["python check.py"],
        max_agent_steps=10,
        prompt_bundle_dir=prompt_dir,
        tool_context_path=tool_context_path,
        plan_output_path=repo / "plan.json",
        candidate_output_path=repo / "candidate.json",
        candidate_plan=CandidatePlan(
            hypothesis="h",
            rationale_summary="r",
            selected_parent_branch="boa/demo/accepted",
            patch_category="optimizer",
            operation_type="replace",
            estimated_risk=0.2,
            informed_by_call_ids=["boa-call-1"],
        ),
    )


def _minimal_config(repo: Path) -> BoaConfig:
    return BoaConfig(
        repo_root=repo,
        boa_md_path=repo / "boa.md",
        config_path=repo / "boa.config",
        run=RunConfig(tag="demo", accepted_branch="boa/demo/accepted"),
        metrics=[MetricConfig(name="accuracy", source="regex", pattern="accuracy=([0-9.]+)")],
        objective=ObjectiveConfig(primary_metric="accuracy", direction="maximize"),
        search=SearchConfig(oracle="bayesian_optimization"),
        runner=RunnerConfig(mode="local", scout=RunnerStageConfig(commands=["echo ok"])),
    )


class CliAgentTests(unittest.TestCase):
    def test_codex_output_schemas_disable_additional_properties(self) -> None:
        self.assertIs(PLAN_SCHEMA["additionalProperties"], False)
        self.assertIs(CANDIDATE_SCHEMA["additionalProperties"], False)
        self.assertEqual(sorted(PLAN_SCHEMA["required"]), sorted(PLAN_SCHEMA["properties"].keys()))
        self.assertEqual(sorted(CANDIDATE_SCHEMA["required"]), sorted(CANDIDATE_SCHEMA["properties"].keys()))

    def test_system_prompts_explain_tools_use_json_stdin(self) -> None:
        repo = _make_repo()
        tool_command = str(repo / "prompts" / "boa-tools")
        planning_prompt = build_planning_system_prompt(repo_root=repo, context=_planning_context(repo), tool_command=tool_command)
        execution_prompt = build_execution_system_prompt(
            repo_root=repo,
            context=_execution_context(repo),
            tool_command=tool_command,
        )

        self.assertIn("Do not invent extra flags such as `--trial-id`", planning_prompt)
        self.assertIn(str(tool_command), planning_prompt)
        self.assertIn("tools list-lineage-options", planning_prompt)
        self.assertIn("Do not invent extra flags such as `--trial-id`", execution_prompt)

    def test_plan_trial_writes_boa_tools_launcher_into_prompt_bundle(self) -> None:
        repo = _make_repo()
        context = _planning_context(repo)
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="custom", runtime="cli", command="python3", args=["-c", "print('{{}}')"]),
        )
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"hypothesis":"h","rationale_summary":"r","selected_parent_branch":"boa/demo/accepted","patch_category":"optimizer","operation_type":"replace","estimated_risk":0.2,"informed_by_call_ids":["boa-call-1"]}',
                stderr="",
            )
            agent.plan_trial(context)
        launcher_path = context.prompt_bundle_dir / "boa-tools"
        launcher_text = launcher_path.read_text(encoding="utf-8")
        planning_prompt = (context.prompt_bundle_dir / "system_prompt.txt").read_text(encoding="utf-8")

        self.assertTrue(launcher_path.exists())
        self.assertIn('exec "', launcher_text)
        self.assertIn("-m boaresearch.cli", launcher_text)
        self.assertIn(str(context.tool_context_path), launcher_text)
        self.assertIn(str(launcher_path), planning_prompt)

    def test_task_prompts_include_explicit_boa_output_paths(self) -> None:
        repo = _make_repo()
        planning_context = _planning_context(repo)
        execution_context = _execution_context(repo)

        planning_task = build_planning_task(planning_context)
        execution_task = build_execution_task(execution_context)

        self.assertIn(str(planning_context.plan_output_path), planning_task)
        self.assertIn("Do not scan `.boa/`", planning_task)
        self.assertIn(str(execution_context.plan_output_path), execution_task)
        self.assertIn(str(execution_context.candidate_output_path), execution_task)
        self.assertIn("Do not scan `.boa/`", execution_task)

    def test_codex_plan_trial_uses_exec_mode_with_schema_output(self) -> None:
        repo = _make_repo()
        context = _planning_context(repo)
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="codex", runtime="cli", command="codex", profile="test-profile", model="gpt-5.4"),
        )
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"hypothesis":"h","rationale_summary":"r","selected_parent_branch":"boa/demo/accepted","patch_category":"optimizer","operation_type":"replace","estimated_risk":0.2,"informed_by_call_ids":["boa-call-1"]}',
                stderr="",
            )
            agent.plan_trial(context)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:3], ["codex", "exec", "-"])
        self.assertIn("--sandbox", command)
        self.assertIn("workspace-write", command)
        self.assertIn("--output-last-message", command)
        self.assertIn("-p", command)
        self.assertIn("test-profile", command)
        self.assertIn("-m", command)
        self.assertIn("gpt-5.4", command)

    def test_codex_prepare_candidate_uses_exec_mode(self) -> None:
        repo = _make_repo()
        context = _execution_context(repo)
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="codex", runtime="cli", command="codex", model="gpt-5.4"),
        )
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"hypothesis":"h","rationale_summary":"r","patch_category":"optimizer","operation_type":"replace","estimated_risk":0.2,"informed_by_call_ids":["boa-call-1"]}',
                stderr="",
            )
            candidate = agent.prepare_candidate(context)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:3], ["codex", "exec", "-"])
        self.assertEqual(candidate.patch_category, "optimizer")

    def test_claude_code_plan_trial_uses_generic_cli_path(self) -> None:
        repo = _make_repo()
        context = _planning_context(repo)
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="claude_code", runtime="cli", command="claude"),
        )
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"hypothesis":"h","rationale_summary":"r","selected_parent_branch":"boa/demo/accepted","patch_category":"optimizer","operation_type":"replace","estimated_risk":0.2,"informed_by_call_ids":["boa-call-2"]}',
                stderr="",
            )
            plan = agent.plan_trial(context)
        command = run_mock.call_args.args[0]
        self.assertEqual(command, ["claude"])
        self.assertEqual(plan.operation_type, "replace")

    def test_copilot_prepare_candidate_reads_output_file(self) -> None:
        repo = _make_repo()
        context = _execution_context(repo)
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="copilot", runtime="cli", command="copilot"),
        )

        def fake_run(*args, **kwargs):
            context.candidate_output_path.write_text(
                json.dumps(
                    {
                        "hypothesis": "h",
                        "rationale_summary": "r",
                        "patch_category": "optimizer",
                        "operation_type": "replace",
                        "estimated_risk": 0.2,
                        "informed_by_call_ids": ["boa-call-3"],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ignored", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as run_mock:
            candidate = agent.prepare_candidate(context)
        command = run_mock.call_args.args[0]
        self.assertEqual(command, ["copilot"])
        self.assertEqual(candidate.informed_by_call_ids, ["boa-call-3"])

    def test_custom_cli_templates_command_args_and_env(self) -> None:
        repo = _make_repo()
        context = _planning_context(repo)
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(
                preset="custom",
                runtime="cli",
                command="{agent}-runner",
                args=["--trial", "{trial_id}", "--out", "{plan_path}"],
                env={"BOA_TEST_MARKER": "{phase}:{trial_id}"},
            ),
        )
        with patch("subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"hypothesis":"h","rationale_summary":"r","selected_parent_branch":"boa/demo/accepted","patch_category":"optimizer","operation_type":"replace","estimated_risk":0.2,"informed_by_call_ids":["boa-call-4"]}',
                stderr="",
            )
            agent.plan_trial(context)
        command = run_mock.call_args.args[0]
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(command[0], "custom-runner")
        self.assertEqual(command[1:5], ["--trial", "demo-0001", "--out", str(context.plan_output_path)])
        self.assertEqual(env["BOA_TEST_MARKER"], "planning:demo-0001")

    def test_cli_agent_streams_live_output_when_observer_is_present(self) -> None:
        repo = _make_repo()
        context = _planning_context(repo)
        observer = RecordingObserver()
        agent = CliResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(
                preset="custom",
                runtime="cli",
                command="python3",
                args=[
                    "-c",
                    (
                        "print('agent-live'); "
                        "print('{{\"hypothesis\":\"h\",\"rationale_summary\":\"r\",\"selected_parent_branch\":\"boa/demo/accepted\","
                        "\"patch_category\":\"optimizer\",\"operation_type\":\"replace\",\"estimated_risk\":0.2,"
                        "\"informed_by_call_ids\":[\"boa-call-5\"]}}')"
                    ),
                ],
            ),
            observer=observer,
        )

        plan = agent.plan_trial(context)

        self.assertEqual(plan.patch_category, "optimizer")
        self.assertTrue(any(event.kind == "process_output" and event.message == "agent-live" for event in observer.events))


class DeepAgentsTests(unittest.TestCase):
    def test_deepagents_plan_trial_submits_candidate_plan(self) -> None:
        repo = _make_repo()
        context = _planning_context(repo)
        captured: dict[str, object] = {}
        agent = DeepAgentsResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="deepagents", runtime="deepagents", backend="ollama", model="qwen2.5-coder:14b"),
            config=_minimal_config(repo),
            run_preflight=lambda: None,
        )

        def fake_build_agent_tools(harness, *, phase):
            del phase
            return {
                "recent_trials": harness.recent_trials,
                "submit_candidate_plan": harness.submit_candidate_plan,
            }

        class FakeDeepAgent:
            def __init__(self, tools):
                self.tools = tools

            def invoke(self, payload):
                captured["payload"] = payload
                call_id = json.loads(self.tools["recent_trials"]())["call_id"]
                self.tools["submit_candidate_plan"](
                    hypothesis="h",
                    rationale_summary="r",
                    selected_parent_branch="boa/demo/accepted",
                    patch_category="optimizer",
                    operation_type="replace",
                    estimated_risk=0.2,
                    informed_by_call_ids=[call_id],
                )
                return {"ok": True}

        with patch("boaresearch.agents.deepagents.build_agent_tools", side_effect=fake_build_agent_tools):
            with patch.object(DeepAgentsResearchAgent, "_build_model", return_value=object()):
                with patch.object(DeepAgentsResearchAgent, "_build_backend", return_value=object()):
                    with patch.object(DeepAgentsResearchAgent, "_create_agent", side_effect=lambda **kwargs: FakeDeepAgent(kwargs["tools"])):
                        plan = agent.plan_trial(context)
        self.assertEqual(plan.patch_category, "optimizer")
        self.assertEqual(captured["payload"]["messages"][0]["role"], "user")

    def test_deepagents_prepare_candidate_runs_preflight_and_submits_candidate(self) -> None:
        repo = _make_repo()
        context = _execution_context(repo)
        preflight_calls: list[str] = []
        agent = DeepAgentsResearchAgent(
            repo_root=repo,
            agent_config=AgentConfig(preset="deepagents", runtime="deepagents", backend="ollama", model="qwen2.5-coder:14b"),
            config=_minimal_config(repo),
            run_preflight=lambda: preflight_calls.append("called"),
        )

        def fake_build_agent_tools(harness, *, phase):
            del phase
            return {
                "recent_trials": harness.recent_trials,
                "run_preflight": harness.run_preflight,
                "submit_candidate": harness.submit_candidate,
            }

        class FakeDeepAgent:
            def __init__(self, tools):
                self.tools = tools

            def invoke(self, payload):
                del payload
                call_id = json.loads(self.tools["recent_trials"]())["call_id"]
                self.tools["run_preflight"]()
                self.tools["submit_candidate"](
                    hypothesis="h",
                    rationale_summary="r",
                    patch_category="optimizer",
                    operation_type="replace",
                    estimated_risk=0.2,
                    informed_by_call_ids=[call_id],
                )
                return {"ok": True}

        with patch("boaresearch.agents.deepagents.build_agent_tools", side_effect=fake_build_agent_tools):
            with patch.object(DeepAgentsResearchAgent, "_build_model", return_value=object()):
                with patch.object(DeepAgentsResearchAgent, "_build_backend", return_value=object()):
                    with patch.object(DeepAgentsResearchAgent, "_create_agent", side_effect=lambda **kwargs: FakeDeepAgent(kwargs["tools"])):
                        candidate = agent.prepare_candidate(context)
        self.assertEqual(candidate.patch_category, "optimizer")
        self.assertEqual(preflight_calls, ["called"])


if __name__ == "__main__":
    unittest.main()
