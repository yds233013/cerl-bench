"""Decision 2 and 3: no chaining, one transition per step, declared exemptions."""

from __future__ import annotations

import pytest

from cerl.actions import ActionKind, Finish, SlackReadThread, SlackRequestApproval
from cerl.core import FrozenMap, LogicalInstant, ScenarioDefect, UserId
from cerl.diff import Origin
from cerl.env import CerlEnv
from cerl.env.responders import schedule_new_firings
from cerl.reference import W2Oracle, oracle_for, run_actions, run_reference
from cerl.scenario import freeze as freeze_module
from cerl.scenario.axes import DEFAULT_AXES
from cerl.scenario.generator import APPROVALS_CHANNEL
from cerl.scenario.responders import (
    EffectKind,
    EffectSpec,
    GuardKind,
    GuardSpec,
    ResponderRule,
    TriggerSpec,
)
from cerl.state import ScheduledFiring


def _obtainable(seed: int = 17):
    axes = FrozenMap({**DEFAULT_AXES.to_dict(), "approval": "missing_obtainable"})
    return freeze_module.materialize(
        "dup_charge_threshold", axes, seed,
        freeze_module.shard_for("dup_charge_threshold", axes, seed),
    )


# --------------------------------------------------------------------------
# no chaining
# --------------------------------------------------------------------------


def test_triggers_see_only_agent_origin_entries(slice_scenario):
    """A rule that would match a responder entry must never fire.

    This is the whole no-chaining guarantee: responder entries are simply not in
    the domain the trigger evaluator looks at.
    """
    episode = run_reference(slice_scenario, W2Oracle())
    assert episode.trace.responder_entries(), "the slice must fire a responder"

    # A rule whose trigger names the responder's own action kind.
    chained = ResponderRule(
        id="r_chained",
        trigger=TriggerSpec(tool="responder"),
        delay_ticks=0,
        effect=(
            EffectSpec(
                kind=EffectKind.POST_MESSAGE,
                channel=APPROVALS_CHANNEL,
                author=UserId(str(slice_scenario.variables["approver"])),
                text="chained",
            ),
        ),
    )
    world = schedule_new_firings(episode.final, (chained,), episode.trace)
    assert world.responder_queue.pending == ()
    assert "r_chained" not in world.responder_queue.fired


def test_agent_entries_view_excludes_responders(slice_scenario):
    trace = run_reference(slice_scenario, W2Oracle()).trace
    assert all(e.origin is Origin.AGENT for e in trace.agent_entries())
    assert len(trace.agent_entries()) < len(trace.entries)


# --------------------------------------------------------------------------
# at most one responder transition per step
# --------------------------------------------------------------------------


def test_only_one_due_firing_is_returned_even_when_several_are_ready(slice_scenario):
    queue = slice_scenario.world.responder_queue
    for rule_id, fire_at in (("r_c", 5), ("r_a", 5), ("r_b", 3)):
        queue = queue.scheduled(
            ScheduledFiring(rule_id=rule_id, fire_at=LogicalInstant(fire_at),
                            triggered_at_step=0),
        )
    due = queue.due(LogicalInstant(10))
    assert due is not None
    # Ordered by (fire_at, rule_id): the earliest, then lexicographic.
    assert (due.rule_id, int(due.fire_at)) == ("r_b", 3)

    queue = queue.consumed(due)
    second = queue.due(LogicalInstant(10))
    assert second is not None
    assert second.rule_id == "r_a"


def test_each_step_appends_at_most_two_entries(all_frozen):
    """One agent entry plus at most one responder entry -- never more."""
    for scenario in all_frozen:
        episode = run_reference(scenario, oracle_for(scenario))
        per_step: dict[int, int] = {}
        for entry in episode.trace.entries:
            per_step[entry.idx] = per_step.get(entry.idx, 0) + 1
        agent_count = len(episode.trace.agent_entries())
        responder_count = len(episode.trace.responder_entries())
        assert responder_count <= agent_count
        assert len(episode.trace.entries) == agent_count + responder_count


def test_responder_effects_are_atomic(slice_scenario):
    """A multi-effect rule is one transition, not one per mutation."""
    episode = run_reference(slice_scenario, W2Oracle())
    grants = [
        e for e in episode.trace.responder_entries()
        if e.responder_rule == "r_manager_grants"
    ]
    assert len(grants) == 1
    # The grant rule posts a message *and* creates an approval: two mutations,
    # one entry.
    assert len(grants[0].action.effect_kinds) == 2
    paths = {op.path.split("/")[1] for op in grants[0].business_diff.ops}
    assert paths == {"slack"}


def test_pending_effects_at_termination_never_fire():
    scenario = _obtainable()
    duplicate = str(scenario.variables["target_charge"])
    approver = UserId(str(scenario.variables["approver"]))
    env = CerlEnv(scenario)
    env.reset()
    env.step(
        SlackRequestApproval(
            channel=APPROVALS_CHANNEL,
            subject_ref=duplicate,
            amount_cents=int(scenario.variables["duplicate_amount"]),
            mentions=(approver,),
        ),
    )
    env.step(Finish(summary="quitting before the approver replies"))
    assert env.done
    # The grant was scheduled but the episode ended first: no approval exists.
    assert not env.world.slack.approvals
    assert env.world.responder_queue.pending


def test_once_rules_do_not_refire_under_spam():
    """T15: spamming approval requests yields one approval and a step-budget cost."""
    scenario = _obtainable()
    duplicate = str(scenario.variables["target_charge"])
    amount = int(scenario.variables["duplicate_amount"])
    approver = UserId(str(scenario.variables["approver"]))
    request = SlackRequestApproval(
        channel=APPROVALS_CHANNEL, subject_ref=duplicate, amount_cents=amount,
        mentions=(approver,),
    )
    env = CerlEnv(scenario)
    env.reset()
    for _ in range(8):
        env.step(request)
    for _ in range(4):
        env.step(SlackReadThread(channel=APPROVALS_CHANNEL))
    granted = [a for a in env.world.slack.approvals.values() if a.subject_ref == duplicate]
    assert len(granted) == 1
    assert len(env.world.trace.responder_entries()) == 1


def test_guard_selects_between_grant_and_denial():
    obtainable = _obtainable()
    axes = FrozenMap({**DEFAULT_AXES.to_dict(), "approval": "missing_unobtainable"})
    unobtainable = freeze_module.materialize(
        "dup_charge_threshold",
        axes,
        17,
        freeze_module.shard_for("dup_charge_threshold", axes, 17),
    )
    for scenario, expected_rule, expect_approval in (
        (obtainable, "r_manager_grants", True),
        (unobtainable, "r_manager_denies", False),
    ):
        episode = run_reference(scenario, oracle_for(scenario))
        fired = {e.responder_rule for e in episode.trace.responder_entries()}
        assert fired == {expected_rule}, scenario.scenario_id
        assert bool(episode.final.slack.approvals) is expect_approval


# --------------------------------------------------------------------------
# decision 3 - declared exemptions only
# --------------------------------------------------------------------------


def test_responder_diffs_carry_origin_and_rule(slice_scenario):
    episode = run_reference(slice_scenario, W2Oracle())
    for entry in episode.trace.responder_entries():
        assert entry.business_diff.ops
        for op in entry.business_diff.ops:
            assert op.origin is Origin.RESPONDER
            assert op.responder_rule == entry.responder_rule


def test_oracle_play_produces_no_undeclared_responder_effects(all_frozen):
    """The CI gate that makes per-branch declaration mandatory in practice."""
    for scenario in all_frozen:
        verdict = run_reference(scenario, oracle_for(scenario)).verdict
        assert not verdict.undeclared_responder_effects, (
            f"{scenario.scenario_id}: undeclared "
            f"{[op.path for op in verdict.undeclared_responder_effects]}"
        )


def test_freeze_refuses_a_branch_that_does_not_declare_a_reachable_rule():
    """T16 at authoring time: the laundering channel is closed before it opens."""
    scenario = _obtainable()
    stripped = tuple(
        spec for spec in scenario.permitted_diffs if spec.rule != "r_manager_grants"
    )
    undeclared = scenario.model_copy(update={"permitted_diffs": stripped})

    from cerl.scenario.freeze import _check_responder_declarations

    with pytest.raises(ScenarioDefect, match="r_manager_grants"):
        _check_responder_declarations(undeclared)


def test_undeclared_responder_effect_surfaces_and_is_not_charged_to_the_agent():
    """T16 at grading time.

    The agent triggers a responder whose diffs this branch does not declare. The
    ops must surface -- otherwise a responder is a laundering channel -- but they
    must land in ``undeclared_responder_effects``, not in the agent's tally: the
    agent cannot author responder rules, so blaming it would be a misattribution.
    """
    scenario = _obtainable()
    stripped = tuple(
        spec for spec in scenario.permitted_diffs if spec.rule != "r_manager_grants"
    )
    undeclared = scenario.model_copy(update={"permitted_diffs": stripped})

    gold = run_reference(scenario, oracle_for(scenario))
    episode = run_actions(undeclared, gold.actions)
    verdict = episode.verdict

    assert verdict.undeclared_responder_effects, "the responder's diffs must surface"
    assert all(
        op.origin is Origin.RESPONDER for op in verdict.undeclared_responder_effects
    )
    assert not any(
        op.responder_rule == "r_manager_grants" for op in verdict.prohibited_side_effects
    )


def test_responder_permit_does_not_cover_the_same_path_for_the_agent():
    """An agent must not inherit a responder's permit for the same path."""
    scenario = _obtainable()
    responder_permits = [
        spec for spec in scenario.permitted_diffs
        if spec.origin is Origin.RESPONDER
    ]
    assert responder_permits
    for spec in responder_permits:
        assert spec.rule is not None
        assert spec.origin is Origin.RESPONDER


def test_guard_reachability_is_precise():
    """Only rules whose guard can hold for this instance need declaring.

    Requiring an author to declare a rule that provably cannot fire would train
    them to declare everything, which is how a per-branch declaration quietly
    becomes the global exemption it replaced.
    """
    from cerl.scenario.freeze import reachable_responder_rules

    obtainable = _obtainable()
    assert reachable_responder_rules(obtainable) == frozenset({"r_manager_grants"})

    axes = FrozenMap({**DEFAULT_AXES.to_dict(), "approval": "missing_unobtainable"})
    unobtainable = freeze_module.materialize(
        "dup_charge_threshold",
        axes,
        17,
        freeze_module.shard_for("dup_charge_threshold", axes, 17),
    )
    assert reachable_responder_rules(unobtainable) == frozenset({"r_manager_denies"})


def test_guard_holds_boundaries():
    at_most = GuardSpec(kind=GuardKind.AMOUNT_AT_MOST, limit_cents=100)
    greater = GuardSpec(kind=GuardKind.AMOUNT_GREATER_THAN, limit_cents=100)
    assert at_most.holds(100) and not at_most.holds(101)
    assert greater.holds(101) and not greater.holds(100)
    assert GuardSpec().holds(10**9)


def test_responder_actions_are_never_agent_actions(w2_frozen):
    for scenario in w2_frozen[:20]:
        for entry in run_reference(scenario, oracle_for(scenario)).trace.responder_entries():
            assert str(entry.action.kind) == "responder"
            assert entry.action_kind not in {k.value for k in ActionKind}
