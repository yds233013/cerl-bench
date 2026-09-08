"""Can a known-correct policy finish inside the protocol's limits?

This is the check the pilot did not do before it ran, and its absence is what
made 6 of the 15 selected scenarios unsolvable by *any* policy. It costs
seconds, needs no model, and would have caught the defect immediately.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import TypeAdapter

from cerl.actions import Action
from cerl.scenario import freeze
from cerl.verify.verifier import verify
from cerl_rl import protocol
from cerl_rl.environment import drive, scripted, terminal_index

GOLD = pathlib.Path(__file__).resolve().parents[2] / "scenarios" / "v2" / "gold"
_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def _gold(scenario_id: str) -> tuple[Action, ...]:
    payload = json.loads((GOLD / f"{scenario_id}.json").read_text())
    return tuple(_ADAPTER.validate_python(a) for a in payload["actions"])


def _safe_at(scenario_id: str, limit: int) -> bool:
    scenario = freeze.load(freeze.FROZEN_DIR / f"{scenario_id}.json")
    env, _actions, _declared = drive(
        scenario, scripted(_gold(scenario_id)), max_actions=limit,
    )
    verdict = verify(scenario, scenario.world, env.world, env.world.trace)
    return bool(verdict.safe_completion)


def _selected() -> list[str]:
    return [
        *protocol.training_selection().scenario_ids,
        *protocol.validation_selection().scenario_ids,
    ]


def test_the_gold_trajectory_solves_every_selected_scenario_when_unconstrained():
    """Anchors the rest: these are genuinely solvable workflows."""
    for scenario_id in _selected():
        assert _safe_at(scenario_id, len(_gold(scenario_id))), scenario_id


@pytest.mark.parametrize("scenario_id", _selected())
def test_known_correct_workflows_fit_the_proposed_limit(scenario_id):
    """The property the next run needs, asserted per scenario."""
    assert _safe_at(scenario_id, protocol.NEXT_MAX_ACTIONS), (
        f"{scenario_id} cannot be solved within NEXT_MAX_ACTIONS="
        f"{protocol.NEXT_MAX_ACTIONS}; the limit is too tight"
    )


def test_the_limit_the_pilot_actually_used_was_infeasible():
    """Pins the defect rather than quietly fixing it.

    Kept as an assertion so the historical result stays interpretable: the
    recorded run's rewards were earned under a limit at which most of the task
    could not be completed at all.
    """
    solvable = [s for s in _selected() if _safe_at(s, protocol.MAX_ACTIONS)]
    assert len(solvable) == 9, len(solvable)
    assert len(_selected()) == 15


def test_the_proposed_limit_leaves_headroom_over_the_reference_solution():
    """Gold length is not a proven minimum, so the limit is not set to it."""
    needed = max(terminal_index(_gold(s)) + 1 for s in _selected())
    assert needed == 16
    assert needed * 1.4 <= protocol.NEXT_MAX_ACTIONS, (
        "the limit should not sit on the reference solution's length; a correct "
        "policy that takes a redundant read must still be able to finish"
    )
