"""Running an evaluated agent against a scenario, with bounded everything.

This lives in ``eval`` rather than ``agents`` for a reason the import contract
surfaced: constructing a ``CerlEnv`` gives its importer a transitive path to
``verify`` and ``scenario``. An evaluated policy must not have that reach even
indirectly, so the wiring between a policy and an environment is evaluation-layer
work and the policy package stays unable to see the verifier at all.
"""

from __future__ import annotations

from cerl.actions import Action
from cerl.agents.base import Agent
from cerl.agents.prompt_only import DEFAULT_MAX_STEPS, AgentTranscript
from cerl.core import Frozen
from cerl.env.env import CerlEnv


class AgentRun(Frozen):
    actions: tuple[Action, ...]
    transcript: AgentTranscript | None = None
    steps: int
    stopped_early: bool
    stop_reason: str


def run_agent(
    env: CerlEnv,
    agent: Agent,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentRun:
    """Drive one episode. Bounded by ``max_steps`` and by the env's own budget."""
    observation = env.reset()
    actions: list[Action] = []
    stop_reason = "declared"

    while not env.done:
        if len(actions) >= max_steps:
            stop_reason = "max_steps"
            break
        action = agent.act(observation)
        actions.append(action)
        observation = env.step(action).observation

    if env.done and stop_reason == "declared":
        stop_reason = "truncated" if not env.declared_outcome else "declared"

    return AgentRun(
        actions=tuple(actions),
        transcript=getattr(agent, "transcript", None),
        steps=len(actions),
        stopped_early=stop_reason != "declared",
        stop_reason=stop_reason,
    )
