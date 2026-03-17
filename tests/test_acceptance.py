from __future__ import annotations

import unittest
from pathlib import Path

from boaresearch.acceptance import AcceptanceEngine
from boaresearch.schema import BoaConfig, MetricConfig, ObjectiveConfig, RunnerConfig, RunnerStageConfig


def build_config(direction: str = "maximize") -> BoaConfig:
    return BoaConfig(
        metrics=[
            MetricConfig(name="accuracy", source="regex", pattern="accuracy=([0-9.]+)"),
            MetricConfig(name="cost", source="regex", pattern="cost=([0-9.]+)", required=False),
        ],
        objective=ObjectiveConfig(
            primary_metric="accuracy",
            direction=direction,
            threshold=0.9 if direction == "maximize" else 0.1,
            cost_penalty_metric="cost",
            cost_penalty_weight=0.5,
            minimum_improvement_delta=0.01,
        ),
        runner=RunnerConfig(mode="local", scout=RunnerStageConfig(commands=["echo ok"])),
        boa_md_path=Path("boa.md"),
    )


class AcceptanceTests(unittest.TestCase):
    def test_maximize_threshold_and_penalty(self) -> None:
        engine = AcceptanceEngine(build_config("maximize"))
        evaluation = engine.evaluate_stage(
            stage_name="scout",
            metrics={"accuracy": 0.95, "cost": 0.2},
            incumbent=None,
        )
        self.assertTrue(evaluation.threshold_passed)
        self.assertTrue(evaluation.improved)
        self.assertAlmostEqual(evaluation.adjusted_score, 0.85)

    def test_minimize_inverts_primary_metric(self) -> None:
        config = build_config("minimize")
        config.objective.primary_metric = "loss"
        config.objective.threshold = 0.2
        config.objective.cost_penalty_metric = None
        config.metrics = [MetricConfig(name="loss", source="regex", pattern="loss=([0-9.]+)")]
        engine = AcceptanceEngine(config)
        evaluation = engine.evaluate_stage(
            stage_name="scout",
            metrics={"loss": 0.15},
            incumbent=None,
        )
        self.assertTrue(evaluation.threshold_passed)
        self.assertAlmostEqual(evaluation.adjusted_score, -0.15)


if __name__ == "__main__":
    unittest.main()
