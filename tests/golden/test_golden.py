"""Committed trajectories with committed verdicts and state hashes.

Any change to reward, diff, projection, taxonomy or predicate semantics breaks
these loudly. Regenerating them requires a deliberate version bump, never a quiet
overwrite -- which is the point: the goldens are what stop grading semantics from
drifting under a published number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cerl.reference import W2Oracle, run_actions, run_reference
from cerl.reference.gold import load as load_gold
from cerl.scenario import freeze as freeze_module
from cerl.verify import VERIFIER_VERSION
from cerl.verify.verifier import predicate_library_hash
from tests.helpers import FROZEN_DIR, GOLD_DIR, frozen_paths

GOLDEN_VERDICTS = Path(__file__).parent / "golden_verdicts.json"


def _verdict_fingerprint(verdict) -> dict:
    return {
        "branch": verdict.branch,
        "required_decision": verdict.required_decision,
        "declared_outcome": verdict.declared_outcome,
        "rubric": dict(verdict.rubric),
        "correct_final_state": verdict.correct_final_state,
        "task_completion": verdict.task_completion,
        "decision_correct": verdict.decision_correct,
        "prohibited_side_effects": len(verdict.prohibited_side_effects),
        "undeclared_responder_effects": len(verdict.undeclared_responder_effects),
        "violations": len(verdict.violations),
        "attempted_violations": len(verdict.attempted_violations),
        "tool_calls": verdict.tool_calls,
        "failure_class": str(verdict.failure_class),
    }


def test_gold_corpus_covers_every_frozen_scenario():
    frozen = {p.stem for p in frozen_paths()}
    gold = {p.stem for p in sorted(GOLD_DIR.glob("*.json"))}
    assert frozen == gold, frozen ^ gold


def test_gold_actions_reproduce_their_recorded_verdicts():
    for path in sorted(GOLD_DIR.glob("*.json")):
        gold = load_gold(path)
        scenario = freeze_module.load(FROZEN_DIR / f"{gold.scenario_id}.json")
        episode = run_actions(scenario, gold.actions)
        assert episode.final.state_hash() == gold.terminal_state_hash, gold.scenario_id
        assert episode.trace.head_hash == gold.trace_head_hash, gold.scenario_id
        assert episode.verdict.tool_calls == gold.tool_calls, gold.scenario_id
        assert episode.verdict.is_clean_oracle_run, gold.scenario_id


def test_gold_records_the_grading_versions_it_was_produced_under():
    for path in sorted(GOLD_DIR.glob("*.json")):
        gold = load_gold(path)
        assert gold.verifier_version == VERIFIER_VERSION, (
            f"{gold.scenario_id} was graded under verifier {gold.verifier_version}; "
            f"current is {VERIFIER_VERSION}. Regenerate the corpus deliberately."
        )
        assert gold.predicate_library_hash == predicate_library_hash(), (
            f"{gold.scenario_id}: the predicate library changed. Grading semantics "
            f"moved -- bump the version and regenerate rather than overwriting."
        )


def test_golden_verdicts_are_unchanged():
    """The tripwire for silent grading drift."""
    if not GOLDEN_VERDICTS.exists():
        pytest.skip("golden verdicts not yet generated")
    expected = json.loads(GOLDEN_VERDICTS.read_text(encoding="utf-8"))
    actual = {}
    for path in frozen_paths():
        scenario = freeze_module.load(path)
        actual[scenario.scenario_id] = _verdict_fingerprint(
            run_reference(scenario, W2Oracle()).verdict,
        )
    assert actual == expected


def test_golden_chain_verifies():
    for path in sorted(GOLD_DIR.glob("*.json"))[:20]:
        gold = load_gold(path)
        scenario = freeze_module.load(FROZEN_DIR / f"{gold.scenario_id}.json")
        run_actions(scenario, gold.actions).trace.verify_chain()
