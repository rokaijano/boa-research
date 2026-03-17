from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import BoaConfig, SearchDecision, TrialSummary


@dataclass
class RepoSearchState:
    accepted_branch: str


class SearchPolicy:
    name = "base"

    def propose(self, memory: list[TrialSummary], repo_state: RepoSearchState, config: BoaConfig) -> SearchDecision:
        raise NotImplementedError


class RandomPolicy(SearchPolicy):
    name = "random"

    def propose(self, memory: list[TrialSummary], repo_state: RepoSearchState, config: BoaConfig) -> SearchDecision:
        rng = random.Random((config.search.seed or 0) + len(memory))
        successful = [trial for trial in memory if trial.canonical_score is not None]
        if not successful:
            return SearchDecision(
                policy=self.name,
                parent_branch=repo_state.accepted_branch,
                prompt_hints=["Explore a new direction from the accepted branch with modest risk."],
                budget_hint="scout",
            )
        chosen = rng.choice(successful)
        hints = ["Explore a fresh variation rather than repeating the last exact patch."]
        if chosen.descriptor:
            hints.append(f"Use {chosen.descriptor.patch_category} as a loose family prior.")
        return SearchDecision(
            policy=self.name,
            parent_branch=chosen.branch_name,
            parent_trial_id=chosen.trial_id,
            patch_category_hint=chosen.descriptor.patch_category if chosen.descriptor else None,
            prompt_hints=hints,
            budget_hint="scout",
        )


class GreedyBestFirstPolicy(SearchPolicy):
    name = "greedy_best_first"

    def propose(self, memory: list[TrialSummary], repo_state: RepoSearchState, config: BoaConfig) -> SearchDecision:
        successful = [trial for trial in memory if trial.canonical_score is not None]
        if not successful:
            return SearchDecision(
                policy=self.name,
                parent_branch=repo_state.accepted_branch,
                prompt_hints=["Start from the accepted branch and prioritize the clearest improvement path."],
                budget_hint="scout",
            )
        chosen = max(successful, key=lambda trial: float(trial.canonical_score or 0.0))
        hints = ["Lean into the strongest prior improvement and keep the patch focused."]
        if chosen.descriptor:
            hints.append(f"Revisit the {chosen.descriptor.patch_category} family with a nearby change.")
            if chosen.descriptor.numeric_knobs:
                rendered = ", ".join(f"{key}={value}" for key, value in sorted(chosen.descriptor.numeric_knobs.items()))
                hints.append(f"Prior successful numeric knobs: {rendered}.")
        return SearchDecision(
            policy=self.name,
            parent_branch=chosen.branch_name,
            parent_trial_id=chosen.trial_id,
            patch_category_hint=chosen.descriptor.patch_category if chosen.descriptor else None,
            prompt_hints=hints,
            budget_hint="scout",
        )


class LocalRankingPolicy(SearchPolicy):
    name = "local_ranking"

    def _rank(self, trial: TrialSummary, config: BoaConfig) -> float:
        score = float(trial.canonical_score or 0.0)
        descriptor = trial.descriptor
        if descriptor is None:
            return score
        if descriptor.numeric_knobs:
            score += 0.05 * min(len(descriptor.numeric_knobs), 5)
        if descriptor.patch_category != "misc":
            score += float(config.search.family_bonus)
        score += float(config.search.lineage_bonus) if trial.parent_trial_id else 0.0
        score -= float(config.search.risk_penalty) * float(descriptor.estimated_risk)
        return score

    def propose(self, memory: list[TrialSummary], repo_state: RepoSearchState, config: BoaConfig) -> SearchDecision:
        successful = [trial for trial in memory if trial.canonical_score is not None]
        if not successful:
            return SearchDecision(
                policy=self.name,
                parent_branch=repo_state.accepted_branch,
                prompt_hints=["Start local search around the accepted branch with a low-risk candidate."],
                budget_hint="scout",
            )
        chosen = max(successful, key=lambda trial: self._rank(trial, config))
        hints = ["Stay local to the strongest branch lineage and make one coherent change family."]
        if chosen.descriptor:
            hints.append(f"Preferred family: {chosen.descriptor.patch_category}.")
            if chosen.descriptor.numeric_knobs:
                rendered = ", ".join(f"{key}={value}" for key, value in sorted(chosen.descriptor.numeric_knobs.items()))
                hints.append(f"Bias toward nearby numeric settings: {rendered}.")
            if chosen.descriptor.touched_files:
                hints.append(f"Useful locality signal: {', '.join(chosen.descriptor.touched_files[:5])}.")
        return SearchDecision(
            policy=self.name,
            parent_branch=chosen.branch_name,
            parent_trial_id=chosen.trial_id,
            patch_category_hint=chosen.descriptor.patch_category if chosen.descriptor else None,
            prompt_hints=hints,
            budget_hint="scout",
        )


def build_search_policy(config: BoaConfig) -> SearchPolicy:
    policy = config.search.policy
    if policy == "random":
        return RandomPolicy()
    if policy == "greedy_best_first":
        return GreedyBestFirstPolicy()
    if policy == "local_ranking":
        return LocalRankingPolicy()
    raise ValueError(f"Unsupported search policy: {policy}")
