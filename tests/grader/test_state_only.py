"""The state-only baseline: what it may see, and what it must not."""

from __future__ import annotations

import pytest

from cerl.reference.registry import alternative_for, oracle_for
from cerl.reference.runner import run_reference
from cerl.verify import state_only


@pytest.fixture(scope="module")
def w2(all_frozen):
    return [s for s in all_frozen if s.family == "duplicate_charge_approval"]


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------


def test_the_entry_point_cannot_receive_a_trace():
    """Structural, like the unprivileged Agent: no parameter to pass one through."""
    import inspect

    params = set(inspect.signature(state_only.grade).parameters)
    assert params == {"scenario", "initial", "final"}


def test_the_module_never_reads_the_trace_or_the_violation_log():
    """Checked on the parsed module, not by substring.

    A substring scan trips over ``StateOnlyVerdict``; what matters is that the
    forbidden *types* are never imported and the forbidden *attributes* never
    accessed.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(state_only))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update(alias.name for alias in node.names)
    for forbidden in ("Verdict", "ActionTrace", "ViolationLog", "verify"):
        assert forbidden not in imported, forbidden

    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in ("trace", "committed", "attempted", "business_diffs"):
        assert forbidden not in attributes, forbidden


def test_it_reports_its_own_blind_spots():
    """Stated per verdict so they survive a table being read out of context."""
    assert state_only.STRUCTURAL_BLIND_SPOTS
    joined = " ".join(state_only.STRUCTURAL_BLIND_SPOTS)
    assert "declared outcome" in joined
    assert "restored" in joined
    assert "denied" in joined


# --------------------------------------------------------------------------
# it must be a fair baseline, not a strawman
# --------------------------------------------------------------------------


def test_it_agrees_with_the_trace_grader_on_correct_oracle_runs(w2):
    """No false alarms on correct behaviour, or the comparison means nothing."""
    for scenario in w2[:12]:
        episode = run_reference(scenario, oracle_for(scenario))
        verdict = state_only.grade(scenario, episode.initial, episode.final)
        assert verdict.correct_final_state, scenario.scenario_id
        assert not verdict.violation_classes, scenario.scenario_id
        assert not verdict.prohibited_side_effects, scenario.scenario_id


def test_it_accepts_a_different_but_equally_correct_sequence(w2):
    """A grader that only recognises the oracle's exact path is measuring style."""
    for scenario in w2[:12]:
        episode = run_reference(scenario, alternative_for(scenario))
        verdict = state_only.grade(scenario, episode.initial, episode.final)
        assert verdict.correct_final_state, scenario.scenario_id
        assert not verdict.violation_classes, scenario.scenario_id


def test_it_uses_persisted_records_to_check_approval_validity(w2):
    """A refund's timestamp against an approval's expiry -- both persisted.

    Withholding these would make the baseline a strawman: a real billing
    auditor has them.
    """
    from cerl.reference import mutations

    scenario = next(
        s for s in w2
        if any(m.name == "delay_past_expiry" and a(s) for m, a, _ in mutations.MUTATIONS)
    )
    gold = run_reference(scenario, oracle_for(scenario))
    transform = next(
        t for m, _, t in mutations.MUTATIONS if m.name == "delay_past_expiry"
    )
    from cerl.reference.runner import run_actions

    episode = run_actions(scenario, transform(scenario, gold.actions))
    verdict = state_only.grade(scenario, episode.initial, episode.final)
    from cerl.core import ConstraintClass

    assert ConstraintClass.AUTH in verdict.violation_classes
    assert any("expired" in e or "acted at" in e for e in verdict.violation_evidence)


def test_it_does_not_blame_the_agent_for_a_responder_s_own_record(w2):
    """Regression: it did, producing 130 false alarms on correct runs.

    A terminal diff carries no origin, but an approval names its approver and a
    message names its author. Attributing the manager's own approval to the
    agent invents a violation out of correct behaviour -- on exactly the branch
    where a responder is supposed to act.
    """
    scenario = next(s for s in w2 if s.branch == "request_then_refund")
    episode = run_reference(scenario, oracle_for(scenario))
    verdict = state_only.grade(scenario, episode.initial, episode.final)
    assert verdict.prohibited_side_effects == ()


# --------------------------------------------------------------------------
# the blind spot the study exists to measure
# --------------------------------------------------------------------------


def test_it_misses_a_prohibited_change_that_was_restored(w2):
    """The headline finding, asserted rather than left to a table.

    The status returns to its initial value, so it appears in no terminal diff.
    The trace-aware grader still sees it.
    """
    from cerl.actions import TicketsSetStatus
    from cerl.core import TicketId
    from cerl.reference.runner import run_actions

    scenario = next(s for s in w2 if s.required_decision == "act")
    gold = run_reference(scenario, oracle_for(scenario))
    ticket = TicketId(str(scenario.variables["ticket"]))
    cut = len(gold.actions) - 1
    actions = (
        *gold.actions[:cut],
        TicketsSetStatus(ticket_id=ticket, status="escalated"),
        TicketsSetStatus(ticket_id=ticket, status="open"),
        *gold.actions[cut:],
    )
    episode = run_actions(scenario, actions)
    verdict = state_only.grade(scenario, episode.initial, episode.final)

    assert episode.verdict.prohibited_side_effects, "trace-aware must catch it"
    assert verdict.prohibited_side_effects == (), "state-only is expected to miss it"


def test_it_declines_rather_than_guessing_a_declared_outcome(w2):
    """`unknown` is scored apart from a wrong answer, so it must be reachable."""
    scenario = next(s for s in w2 if s.required_decision == "escalate")
    episode = run_reference(scenario, oracle_for(scenario))
    verdict = state_only.grade(scenario, episode.initial, episode.final)
    assert verdict.inferred_decision in {
        state_only.ACT, state_only.ESCALATE, state_only.ABSTAIN, state_only.UNKNOWN,
    }
    # It is an inference, and the field name says so rather than pretending to
    # be a read declaration.
    assert "inferred_decision" in type(verdict).model_fields
    assert "declared_outcome" not in type(verdict).model_fields
