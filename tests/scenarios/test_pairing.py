"""Criterion 41 across every family, under both reference policies.

The registered difficulty metric is the **oracle**'s tool-call count: it is what
``FrozenScenario.oracle_tool_calls`` records and what the criterion is evaluated
on. The alternative policy is measured too and reported, because a criterion
that only holds for one of two equally correct strategies would be a property of
that strategy rather than of the scenario pair.
"""

from __future__ import annotations

from cerl.reference import alternative_for, oracle_for, run_reference
from cerl.scenario import siblings
from cerl.scenario.siblings import differing_axes

MAX_CALL_DELTA = 1
MAX_BRIEF_DELTA = 0.15


def declared_pairs(scenarios):
    """Every held-out instance paired with its declared ID sibling."""
    by_key = {
        (s.template_id, tuple(sorted(s.axes.items())), s.root_seed): s for s in scenarios
    }
    pairs = []
    for scenario in scenarios:
        if not siblings.is_held_out(scenario.axes, scenario.template_id):
            continue
        sibling_axes = siblings.sibling_axes(scenario.axes, scenario.template_id)
        key = (scenario.template_id, tuple(sorted(sibling_axes.items())), scenario.root_seed)
        assert key in by_key, (
            f"{scenario.scenario_id} has no frozen ID sibling; the difficulty "
            f"invariant would silently skip it"
        )
        pairs.append((scenario, by_key[key]))
    return pairs


def test_every_family_declares_a_pairing(all_frozen):
    from cerl.scenario.families import registry

    for template_id in registry.template_ids():
        pairing = siblings.pairing_for(template_id)
        assert pairing.interventions
        for intervention in pairing.interventions:
            assert intervention.id_values
            assert intervention.held_out
            # Every sibling target must itself be in-distribution.
            for target in intervention.held_out.values():
                assert target in intervention.id_values


def test_every_held_out_instance_has_exactly_one_sibling(all_frozen):
    pairs = declared_pairs(all_frozen)
    assert pairs
    held_out = [
        s for s in all_frozen if siblings.is_held_out(s.axes, s.template_id)
    ]
    assert len(pairs) == len(held_out)


def test_pairs_differ_only_in_one_declared_intervention_axis(all_frozen):
    for cf, sibling in declared_pairs(all_frozen):
        differing = differing_axes(cf.axes, sibling.axes)
        assert len(differing) == 1, (cf.scenario_id, differing)
        axis = differing[0]
        assert axis in siblings.pairing_for(cf.template_id).axis_names()
        assert cf.root_seed == sibling.root_seed
        assert cf.template_id == sibling.template_id


def test_criterion_41_holds_for_every_family(all_frozen):
    """|delta oracle tool calls| <= 1 for every declared pair, no exemptions."""
    failures = []
    for cf, sibling in declared_pairs(all_frozen):
        assert cf.oracle_tool_calls is not None
        assert sibling.oracle_tool_calls is not None
        delta = abs(cf.oracle_tool_calls - sibling.oracle_tool_calls)
        if delta > MAX_CALL_DELTA:
            failures.append(
                f"{cf.family}: {cf.axes} vs {sibling.axes} seed {cf.root_seed}: "
                f"{cf.oracle_tool_calls} vs {sibling.oracle_tool_calls}",
            )
    assert not failures, "criterion 41 violated:\n" + "\n".join(failures)


def test_criterion_41_also_holds_under_the_alternative_policy(all_frozen):
    """Reported as well as the registered metric.

    A criterion that held only for the oracle would be a property of that
    trajectory rather than of the pair.
    """
    failures = []
    for cf, sibling in declared_pairs(all_frozen):
        cf_calls = run_reference(cf, alternative_for(cf)).verdict.tool_calls
        id_calls = run_reference(sibling, alternative_for(sibling)).verdict.tool_calls
        if abs(cf_calls - id_calls) > MAX_CALL_DELTA:
            failures.append(
                f"{cf.family}: alternative {cf_calls} vs {id_calls} "
                f"({cf.scenario_id})",
            )
    assert not failures, "alternative-policy pairing:\n" + "\n".join(failures[:10])


def test_the_registered_metric_is_the_oracle(all_frozen):
    """oracle_tool_calls must be the oracle's count, not the alternative's."""
    for scenario in all_frozen[:20]:
        episode = run_reference(scenario, oracle_for(scenario))
        assert scenario.oracle_tool_calls == episode.verdict.tool_calls


def test_pairs_share_entity_cardinality_and_brief_length(all_frozen):
    for cf, sibling in declared_pairs(all_frozen):
        assert len(cf.world.billing.customers) == len(sibling.world.billing.customers)
        assert len(cf.world.billing.charges) == len(sibling.world.billing.charges)
        assert len(cf.world.tickets.tickets) == len(sibling.world.tickets.tickets)
        delta = abs(len(cf.brief) - len(sibling.brief)) / max(len(sibling.brief), 1)
        assert delta <= MAX_BRIEF_DELTA, (cf.scenario_id, delta)


def test_no_family_is_silently_omitted(all_frozen):
    """Every family must contribute pairs, or the criterion covers less than it claims."""
    families = {cf.family for cf, _ in declared_pairs(all_frozen)}
    assert families == {
        "duplicate_charge_approval",
        "duplicate_billing_profile",
        "suspicious_refund_escalation",
    }


def test_stratified_axes_are_present_on_both_sides_of_the_split(all_frozen):
    """An axis excluded from pairing must appear in every partition.

    W1's ``identity_evidence`` selects which workflow is correct rather than
    perturbing a fixed one, so it is stratified instead of held out. That is only
    sound if its values really do appear on both sides.
    """
    for template_id, axes in siblings.STRATIFIED_AXES.items():
        scenarios = [s for s in all_frozen if s.template_id == template_id]
        assert scenarios
        for axis in axes:
            values = {s.axes[axis] for s in scenarios}
            assert len(values) >= 2, (template_id, axis, values)
            # And present alongside both ID and held-out intervention values.
            for value in values:
                subset = [s for s in scenarios if s.axes[axis] == value]
                assert any(
                    siblings.is_held_out(s.axes, template_id) for s in subset
                ), (axis, value, "no held-out instance")
                assert any(
                    not siblings.is_held_out(s.axes, template_id) for s in subset
                ), (axis, value, "no in-distribution instance")


# --------------------------------------------------------------------------
# the identity-evidence challenge set is reported apart, never as a pass
# --------------------------------------------------------------------------


def test_stratified_axes_form_no_criterion_41_pairs(all_frozen):
    """A stratified axis must contribute to no matched-pair result.

    Stratification does not make its values equally difficult -- W1's identity
    values cost 12/10/9 oracle calls. It only ensures no pair is formed across
    that mismatch. If one ever were, the criterion would silently start
    reporting a difficulty difference as a generalization gap.
    """
    for template_id, axes in siblings.STRATIFIED_AXES.items():
        pairing = siblings.pairing_for(template_id)
        for axis in axes:
            assert axis not in pairing.axis_names(), (
                f"{template_id}: {axis} is stratified and must not also be an "
                f"intervention axis"
            )
    for cf, sibling in declared_pairs(all_frozen):
        stratified = siblings.STRATIFIED_AXES.get(cf.template_id, ())
        for axis in stratified:
            assert cf.axes[axis] == sibling.axes[axis], (
                f"{cf.scenario_id}: pair varies the stratified axis {axis}"
            )


def test_the_identity_evidence_challenge_set_is_preserved_and_measured(all_frozen):
    """The scenarios stay, and their real difficulty spread is asserted.

    They are kept because they test the family's core hazard, and measured
    because pretending they are matched would be worse than saying they are not.
    """
    w1 = [s for s in all_frozen if s.template_id == "dup_profile_reconciliation"]
    values = {s.axes["identity_evidence"] for s in w1}
    assert values == {"strong_match", "conflicting_external_ref", "name_only_similarity"}

    baseline = {
        s.axes["identity_evidence"]: s.oracle_tool_calls
        for s in w1
        if s.axes["merge_approval"] == "valid"
        and s.axes["dispute_state"] == "none"
        and s.axes["tool_reliability"] == "stable"
        and s.axes["prior_progress"] == "none"
    }
    assert baseline == {
        "strong_match": 12,
        "name_only_similarity": 10,
        "conflicting_external_ref": 9,
    }, baseline

    # The spread exceeds +/-1, which is exactly why this is not a matched set.
    reference = baseline["strong_match"]
    spread = {k: abs(v - reference) for k, v in baseline.items()}
    assert spread["name_only_similarity"] == 2
    assert spread["conflicting_external_ref"] == 3

    # Structural parity does hold, so the difference is interaction cost alone.
    by_value = {}
    for scenario in w1:
        by_value.setdefault(scenario.axes["identity_evidence"], scenario)
    sizes = {
        (len(s.world.billing.customers), len(s.world.billing.charges), len(s.brief))
        for s in by_value.values()
    }
    assert len(sizes) == 1, sizes
