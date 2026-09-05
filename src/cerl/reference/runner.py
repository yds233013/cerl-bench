"""Run a policy against a scenario and score it.

Two entry points, deliberately distinct: only ``run_reference`` can supply a
``GroundTruthView``. An unprivileged agent physically cannot receive one, because
``Agent.act`` has no parameter to receive it through.
"""

from __future__ import annotations

from cerl.actions import Action
from cerl.core import Frozen
from cerl.env import CerlEnv
from cerl.env.observation import Observation
from cerl.reference.ground_truth import GroundTruthView, ReferencePolicy, ground_truth_for
from cerl.scenario.schema import FrozenScenario
from cerl.state import WorldState
from cerl.trace import ActionTrace
from cerl.verify import Verdict, verify


class Episode(Frozen):
    scenario_id: str
    actions: tuple[Action, ...]
    initial: WorldState
    final: WorldState
    trace: ActionTrace
    truncated: bool
    verdict: Verdict

    @property
    def tool_calls(self) -> int:
        return self.verdict.tool_calls


def _finish_episode(
    scenario: FrozenScenario,
    env: CerlEnv,
    actions: list[Action],
    truncated: bool,  # noqa: FBT001
    oracle_tool_calls: int | None,
) -> Episode:
    verdict = verify(
        scenario,
        scenario.world,
        env.world,
        env.world.trace,
        truncated=truncated,
        oracle_tool_calls=oracle_tool_calls,
    )
    return Episode(
        scenario_id=scenario.scenario_id,
        actions=tuple(actions),
        initial=scenario.world,
        final=env.world,
        trace=env.world.trace,
        truncated=truncated,
        verdict=verdict,
    )


def run_reference(
    scenario: FrozenScenario,
    policy: ReferencePolicy,
    *,
    oracle_tool_calls: int | None = None,
) -> Episode:
    """Run a privileged reference policy (the oracle)."""
    env = CerlEnv(scenario)
    observation: Observation = env.reset()
    truth: GroundTruthView = ground_truth_for(scenario)
    actions: list[Action] = []

    while not env.done:
        action = policy.act(observation, truth)
        actions.append(action)
        result = env.step(action)
        observation = result.observation

    truncated = env.world.meta.step_index >= scenario.budget_steps
    return _finish_episode(scenario, env, actions, truncated, oracle_tool_calls)


def run_actions(
    scenario: FrozenScenario,
    actions: tuple[Action, ...],
    *,
    oracle_tool_calls: int | None = None,
) -> Episode:
    """Run a fixed action list. Used by golden, adversarial and mutation tests."""
    env = CerlEnv(scenario)
    env.reset()
    played: list[Action] = []
    for action in actions:
        if env.done:
            break
        played.append(action)
        env.step(action)
    # An unterminated script is scored as incomplete, never auto-finished:
    # silently appending a terminal action would hide the failure.
    truncated = env.world.meta.step_index >= scenario.budget_steps
    return _finish_episode(scenario, env, played, truncated, oracle_tool_calls)

