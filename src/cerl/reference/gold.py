"""Gold trajectories.

Committed alongside the frozen scenarios so that:

* every scenario is provably solvable (the oracle reached a clean 1.0);
* replay has something to verify against in CI;
* the efficiency denominator (``oracle_tool_calls``) is a recorded fact rather
  than a guess.

Lives in ``reference`` because producing it requires ground truth. Nothing in
``agents`` can reach this module (import-linter contract 1, plus a reachability
test).
"""

from __future__ import annotations

import json
from pathlib import Path

from cerl.actions import Action
from cerl.core import Frozen
from cerl.reference.ground_truth import ReferencePolicy
from cerl.reference.registry import oracle_for as _oracle_for
from cerl.reference.runner import Episode, run_reference
from cerl.scenario import corpus
from cerl.scenario.schema import FrozenScenario

GOLD_DIR = corpus.CANONICAL.gold


class GoldTrajectory(Frozen):
    scenario_id: str
    branch: str
    required_decision: str
    actions: tuple[Action, ...]
    tool_calls: int
    terminal_state_hash: str
    trace_head_hash: str
    verifier_version: str
    predicate_library_hash: str


def oracle_for(scenario: FrozenScenario) -> ReferencePolicy:
    """The registered reference policy for this scenario's family."""
    return _oracle_for(scenario)


def produce(scenario: FrozenScenario) -> tuple[Episode, GoldTrajectory]:
    """Run the oracle and package its trajectory."""
    episode = run_reference(scenario, oracle_for(scenario))
    verdict = episode.verdict
    gold = GoldTrajectory(
        scenario_id=scenario.scenario_id,
        branch=scenario.branch,
        required_decision=scenario.required_decision,
        actions=episode.actions,
        tool_calls=verdict.tool_calls,
        terminal_state_hash=episode.final.state_hash(),
        trace_head_hash=episode.trace.head_hash,
        verifier_version=verdict.verifier_version,
        predicate_library_hash=verdict.predicate_library_hash,
    )
    return episode, gold


def write(gold: GoldTrajectory, directory: Path = GOLD_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{gold.scenario_id}.json"
    payload = gold.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load(path: Path) -> GoldTrajectory:
    return GoldTrajectory.model_validate(json.loads(path.read_text(encoding="utf-8")))
