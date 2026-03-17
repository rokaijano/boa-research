from __future__ import annotations

from dataclasses import dataclass

from .schema import BoaConfig, IncumbentRecord, StageEvaluation


def enabled_stage_order(cfg: BoaConfig) -> list[str]:
    stages = ["scout"]
    if cfg.runner.confirm.enabled:
        stages.append("confirm")
    if cfg.runner.promoted.enabled:
        stages.append("promoted")
    return stages


@dataclass
class AcceptanceEngine:
    config: BoaConfig

    def stage_order(self) -> list[str]:
        return enabled_stage_order(self.config)

    def highest_enabled_stage(self) -> str:
        return self.stage_order()[-1]

    def adjusted_score(self, metrics: dict[str, float]) -> tuple[float, float, float | None]:
        primary_name = self.config.objective.primary_metric
        if primary_name not in metrics:
            raise ValueError(f"Missing primary metric '{primary_name}'")
        primary_metric = float(metrics[primary_name])
        oriented = primary_metric if self.config.objective.direction == "maximize" else -primary_metric
        cost_metric = None
        penalty = 0.0
        if self.config.objective.cost_penalty_metric:
            cost_name = self.config.objective.cost_penalty_metric
            if cost_name not in metrics:
                raise ValueError(f"Missing cost penalty metric '{cost_name}'")
            cost_metric = float(metrics[cost_name])
            penalty = float(self.config.objective.cost_penalty_weight) * cost_metric
        return oriented - penalty, primary_metric, cost_metric

    def threshold_passed(self, primary_metric: float) -> bool:
        threshold = self.config.objective.threshold
        if threshold is None:
            return True
        if self.config.objective.direction == "maximize":
            return primary_metric >= float(threshold)
        return primary_metric <= float(threshold)

    def evaluate_stage(
        self,
        *,
        stage_name: str,
        metrics: dict[str, float],
        incumbent: IncumbentRecord | None,
    ) -> StageEvaluation:
        adjusted_score, primary_metric, cost_metric = self.adjusted_score(metrics)
        threshold_passed = self.threshold_passed(primary_metric)
        if incumbent is None:
            improved = True
        else:
            improved = adjusted_score >= (float(incumbent.adjusted_score) + self.config.objective.minimum_improvement_delta)
        advanced = threshold_passed and improved
        final_accept = advanced and stage_name == self.highest_enabled_stage()
        reason_parts: list[str] = []
        if threshold_passed:
            reason_parts.append("threshold_passed")
        else:
            reason_parts.append("threshold_failed")
        if improved:
            reason_parts.append("improved")
        else:
            reason_parts.append("not_improved")
        if final_accept:
            reason_parts.append("final_accept")
        elif advanced:
            reason_parts.append("advance")
        return StageEvaluation(
            stage_name=stage_name,
            primary_metric=primary_metric,
            adjusted_score=adjusted_score,
            threshold_passed=threshold_passed,
            improved=improved,
            advanced=advanced,
            final_accept=final_accept,
            reason=",".join(reason_parts),
            cost_metric=cost_metric,
        )
