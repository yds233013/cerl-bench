"""Run the deterministic control baselines and report by family and branch."""

from __future__ import annotations

import collections
from collections.abc import Callable

from cerl.agents.base import Agent
from cerl.agents.controls import (
    AlwaysAbstainAgent,
    AlwaysEscalateAgent,
    AlwaysFinishAgent,
    InvestigateThenEscalateAgent,
    RandomValidAgent,
)
from cerl.core import Frozen, FrozenMap
from cerl.env.env import CerlEnv
from cerl.eval.agent_harness import run_agent
from cerl.reference.registry import alternative_for, oracle_for
from cerl.reference.runner import Episode, run_actions, run_reference
from cerl.scenario.schema import FrozenScenario

#: Deterministic controls, plus the two privileged reference policies. The
#: reference rows are labelled privileged: they read ground truth, so they are
#: an upper bound and not a baseline arm.
CONTROLS: dict[str, Callable[[int], Agent]] = {
    "always_escalate": lambda _seed: AlwaysEscalateAgent(),
    "always_abstain": lambda _seed: AlwaysAbstainAgent(),
    "always_finish": lambda _seed: AlwaysFinishAgent(),
    "investigate_then_escalate": lambda _seed: InvestigateThenEscalateAgent(),
    "random_valid": RandomValidAgent,
}

PRIVILEGED = ("oracle[privileged]", "alternative[privileged]")


class ControlResult(Frozen):
    policy: str
    privileged: bool
    episodes: int
    safe_completions: int
    correct_decisions: int
    committed_violation_episodes: int
    attempted_violation_episodes: int
    side_effect_episodes: int
    by_branch_safe: FrozenMap[str, int]
    by_branch_total: FrozenMap[str, int]
    by_family_safe: FrozenMap[str, int]
    by_family_total: FrozenMap[str, int]

    @property
    def safe_rate(self) -> float:
        return self.safe_completions / self.episodes if self.episodes else 0.0

    @property
    def violation_rate(self) -> float:
        return (
            self.committed_violation_episodes / self.episodes if self.episodes else 0.0
        )


def _score(scenario: FrozenScenario, policy: str, seed: int) -> Episode:
    if policy == "oracle[privileged]":
        return run_reference(scenario, oracle_for(scenario))
    if policy == "alternative[privileged]":
        return run_reference(scenario, alternative_for(scenario))
    agent = CONTROLS[policy](seed)
    run = run_agent(CerlEnv(scenario), agent)
    return run_actions(scenario, run.actions)


def evaluate_control(
    policy: str, scenarios: list[FrozenScenario], seed: int = 17,
) -> ControlResult:
    safe = correct = committed = attempted = side_effects = 0
    branch_safe: collections.Counter[str] = collections.Counter()
    branch_total: collections.Counter[str] = collections.Counter()
    family_safe: collections.Counter[str] = collections.Counter()
    family_total: collections.Counter[str] = collections.Counter()

    for scenario in scenarios:
        verdict = _score(scenario, policy, seed).verdict
        key = f"{scenario.family}/{scenario.branch}"
        branch_total[key] += 1
        family_total[scenario.family] += 1
        if verdict.safe_completion:
            safe += 1
            branch_safe[key] += 1
            family_safe[scenario.family] += 1
        correct += int(verdict.decision_correct)
        committed += int(bool(verdict.violations))
        attempted += int(bool(verdict.attempted_violations))
        side_effects += int(bool(verdict.prohibited_side_effects))

    return ControlResult(
        policy=policy,
        privileged=policy in PRIVILEGED,
        episodes=len(scenarios),
        safe_completions=safe,
        correct_decisions=correct,
        committed_violation_episodes=committed,
        attempted_violation_episodes=attempted,
        side_effect_episodes=side_effects,
        by_branch_safe=FrozenMap(dict(branch_safe)),
        by_branch_total=FrozenMap(dict(branch_total)),
        by_family_safe=FrozenMap(dict(family_safe)),
        by_family_total=FrozenMap(dict(family_total)),
    )


def all_policies() -> tuple[str, ...]:
    return (*PRIVILEGED, *sorted(CONTROLS))
