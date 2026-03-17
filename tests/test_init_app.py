from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boaresearch.init_app import InitWizard, PromptAdapter
from boaresearch.init_models import (
    DetectedRepo,
    ExistingSetupReport,
    InitSetupSelection,
    PreflightCheck,
    RepoAnalysisProposal,
    ReviewedInitPlan,
    ValidationReport,
    WriteResult,
)
from boaresearch.init_services import InitServices


class FakePrompts:
    def __init__(self, answers: list[object]) -> None:
        self.answers = list(answers)

    def _next(self):
        if not self.answers:
            raise AssertionError("No prompt answers left")
        return self.answers.pop(0)

    def select(self, **kwargs) -> str:
        return str(self._next())

    def text(self, **kwargs) -> str:
        return str(self._next())

    def confirm(self, **kwargs) -> bool:
        return bool(self._next())


class InitAppTests(unittest.TestCase):
    def _services(self, repo_root: Path) -> InitServices:
        detected = DetectedRepo(
            requested_path=repo_root,
            repo_root=repo_root,
            config_path=repo_root / "boa.config",
            boa_md_path=repo_root / "boa.md",
            runtime_root=repo_root / ".boa",
            existing_setup=ExistingSetupReport(status="absent", config_path=repo_root / "boa.config", boa_md_path=repo_root / "boa.md"),
        )
        selection = InitSetupSelection(repo_root=repo_root, agent_preset="codex", agent_command="codex")
        analysis = RepoAnalysisProposal(
            train_command="python train.py",
            eval_command="python eval.py",
            primary_metric_name="accuracy",
            metric_direction="maximize",
            metric_source="regex",
            metric_path=None,
            metric_json_key=None,
            metric_pattern="accuracy=([0-9.]+)",
            editable_files=["src", "tests"],
            protected_files=[".boa"],
            optimization_surfaces=["src"],
            caveats=["Confirm metric extraction"],
            suggested_boa_md="# BOA Repo Contract",
        )
        plan = ReviewedInitPlan(
            repo_root=repo_root,
            selection=selection,
            analysis=analysis,
            editable_files=["src", "tests"],
            protected_files=[".boa", ".git"],
            train_command=analysis.train_command,
            eval_command=analysis.eval_command,
            primary_metric_name=analysis.primary_metric_name,
            metric_direction=analysis.metric_direction,
            metric_source=analysis.metric_source,
            metric_path=analysis.metric_path,
            metric_json_key=analysis.metric_json_key,
            metric_pattern=analysis.metric_pattern,
            boa_md=analysis.suggested_boa_md,
        )
        return InitServices(
            detect_repo=lambda path: detected,
            default_selection_for_repo=lambda detected_repo: selection,
            run_preflight=lambda detected_repo, setup: [PreflightCheck("git", True, "git found")],
            analyze_repo=lambda detected_repo, setup: analysis,
            default_repo_analysis=lambda root: analysis,
            merge_reviewed_plan=lambda setup, proposal: plan,
            write_contract_files=lambda reviewed: WriteResult(created_paths=[repo_root / "boa.config", repo_root / "boa.md"]),
            validate_written_setup=lambda reviewed: ValidationReport(passed=True, details=["Config parses successfully."]),
        )

    def test_inquirer_flow_reaches_completion(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        prompts = FakePrompts(
            [
                "codex",
                False,
                "default",
                "default",
                "local",
                "",
                False,
                True,
            ]
        )
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=self._services(repo_root),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        draft = wizard.run()
        self.assertIsNotNone(draft.write_result)
        self.assertIsNotNone(draft.validation)
        self.assertTrue(any("No BOA files detected" in line for line in output))

    def test_review_mode_stops_before_writing(self) -> None:
        repo_root = Path(tempfile.mkdtemp())
        services = self._services(repo_root)
        detected = services.detect_repo(repo_root)
        detected.existing_setup = ExistingSetupReport(
            status="valid",
            config_path=repo_root / "boa.config",
            boa_md_path=repo_root / "boa.md",
            summary="Current setup found",
        )
        prompts = FakePrompts(["review"])
        output: list[str] = []
        wizard = InitWizard(
            initial_path=repo_root,
            services=InitServices(
                detect_repo=lambda path: detected,
                default_selection_for_repo=services.default_selection_for_repo,
                run_preflight=services.run_preflight,
                analyze_repo=services.analyze_repo,
                default_repo_analysis=services.default_repo_analysis,
                merge_reviewed_plan=services.merge_reviewed_plan,
                write_contract_files=services.write_contract_files,
                validate_written_setup=services.validate_written_setup,
            ),
            prompts=PromptAdapter(select=prompts.select, text=prompts.text, confirm=prompts.confirm),
            output=output.append,
        )
        draft = wizard.run()
        self.assertIsNone(draft.write_result)


if __name__ == "__main__":
    unittest.main()
