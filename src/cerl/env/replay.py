"""Exact replay.

The strongest determinism guarantee in the project: replaying the agent-origin
actions of a trace must reproduce the entire trace byte-for-byte -- including
every responder entry at its exact logical time and chain position, and every
intermediate state hash.

This is also what makes offline manifest verification possible: given these
actions, these are the scores, with no model in the loop.
"""

from __future__ import annotations

from collections.abc import Sequence

from cerl.actions import Action
from cerl.core import ReplayDivergence
from cerl.diff import Origin
from cerl.env.env import CerlEnv
from cerl.scenario.schema import FrozenScenario
from cerl.state import WorldState
from cerl.trace import ActionTrace


def agent_actions(trace: ActionTrace) -> tuple[Action, ...]:
    return tuple(e.action for e in trace.entries if e.origin is Origin.AGENT)  # type: ignore[misc]


def run_actions(scenario: FrozenScenario, actions: Sequence[Action]) -> WorldState:
    env = CerlEnv(scenario)
    env.reset()
    for action in actions:
        if env.done:
            break
        env.step(action)
    return env.world


def replay(scenario: FrozenScenario, trace: ActionTrace) -> WorldState:
    """Re-run ``trace``'s agent actions and assert byte-identical reproduction."""
    world = run_actions(scenario, agent_actions(trace))
    original = trace.entries
    replayed = world.trace.entries

    if len(original) != len(replayed):
        raise ReplayDivergence(
            f"entry count {len(replayed)} != original {len(original)}",
        )
    for position, (want, got) in enumerate(zip(original, replayed, strict=True)):
        if want.entry_hash != got.entry_hash:
            raise ReplayDivergence(
                f"entry {position} ({want.origin}/{want.action_kind}): "
                f"hash {got.entry_hash[:12]} != {want.entry_hash[:12]}",
            )
    return world
