from __future__ import annotations

import io
import json
import subprocess
import tempfile
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from boaresearch.schema import CandidateMetadata, CandidatePlan
from boaresearch.search import SearchToolContext, write_search_tool_context
from boaresearch.runtime import ExperimentStore
from boaresearch.runtime.tools import run_tools_command


class ToolsCliTests(unittest.TestCase):
    def test_run_tools_command_emits_json_and_call_id(self) -> None:
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "boa.md").write_text("# BOA\n", encoding="utf-8")
        (repo / "boa.config").write_text(
            textwrap.dedent(
                """
                schema_version = 3

                [run]
                tag = "demo"

                [agent]
                preset = "codex"
                runtime = "cli"
                command = "codex"

                [guardrails]
                allowed_paths = ["src"]
                protected_paths = [".boa"]

                [runner]
                mode = "local"

                [runner.local]

                [runner.scout]
                enabled = true
                commands = ["echo ok"]

                [[metrics]]
                name = "accuracy"
                source = "regex"
                pattern = "accuracy=([0-9.]+)"

                [objective]
                primary_metric = "accuracy"
                direction = "maximize"

                [search]
                oracle = "bayesian_optimization"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        store = ExperimentStore((repo / ".boa" / "store" / "experiments.sqlite").resolve())
        store.ensure_schema()
        store.create_trial(
            run_tag="demo",
            trial_id="demo-0001",
            branch_name="boa/demo/trial/demo-0001",
            parent_branch="boa/demo/accepted",
            parent_trial_id=None,
            candidate_plan=CandidatePlan(
                hypothesis="h",
                rationale_summary="plan",
                selected_parent_branch="boa/demo/accepted",
                patch_category="optimizer",
                operation_type="replace",
                estimated_risk=0.2,
                informed_by_call_ids=["boa-call-seed"],
            ),
            candidate=CandidateMetadata(
                hypothesis="h",
                rationale_summary="r",
                patch_category="optimizer",
                operation_type="replace",
                estimated_risk=0.2,
                informed_by_call_ids=["boa-call-seed"],
            ),
            descriptor=None,
            search_trace=[],
            diff_path=None,
            acceptance_status="accepted",
        )
        trace_path = repo / ".boa" / "artifacts" / "trials" / "demo-0002" / "search_calls.jsonl"
        context_path = repo / ".boa" / "prompts" / "demo-0002" / "planning" / "tool_context.json"
        write_search_tool_context(
            context_path,
            SearchToolContext(
                repo_root=repo,
                config_path=repo / "boa.config",
                run_tag="demo",
                trial_id="demo-0002",
                accepted_branch="boa/demo/accepted",
                phase="planning",
                trace_path=trace_path,
            ),
        )

        stdout = io.StringIO()
        with patch("sys.stdin", io.StringIO("{}")):
            with patch("sys.stdout", stdout):
                run_tools_command(
                    Namespace(
                        tool_command="recent-trials",
                        repo=None,
                        config=None,
                        context=context_path,
                    )
                )

        payload = json.loads(stdout.getvalue())
        self.assertIn("call_id", payload)
        self.assertEqual(payload["trials"][0]["trial_id"], "demo-0001")
        self.assertTrue(trace_path.exists())


if __name__ == "__main__":
    unittest.main()
