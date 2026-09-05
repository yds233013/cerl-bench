"""Registered holdouts versus actual training data.

The check that found the leak. It is retained after the 1.2.0 repair and
retargeted at ``partition_v1_0_0``, so the historical finding stays asserted
rather than becoming folklore: under 1.0.0, 85 registered counterfactuals sat in
the training partition. Assertions about the *canonical* split live in
``test_canonical_split.py``.
"""

from __future__ import annotations

import pytest

from cerl.eval import splits
from cerl.scenario import siblings


@pytest.fixture(scope="module")
def inventory(all_frozen):
    return splits.inventory(list(all_frozen))


# --------------------------------------------------------------------------
# the finding, asserted so it cannot silently change
# --------------------------------------------------------------------------


def test_the_1_0_0_train_partition_did_contain_registered_held_out_values(all_frozen):
    """The historical defect, pinned against the frozen 1.0.0 rule.

    Not a bug in the partition -- a consequence of its approved rule, which puts
    a counterfactual and its ID sibling together on purpose. It *was* a bug in
    treating that partition as training data, which is what 1.2.0 fixed.
    """
    train = [
        s for s in all_frozen if splits.partition_v1_0_0(s) is splits.Partition.TRAIN
    ]
    held_out = [s for s in train if siblings.is_held_out(s.axes, s.template_id)]
    assert len(held_out) == 85, len(held_out)
    assert len(train) == 144


def test_training_eligible_scenarios_contain_no_registered_held_out_value(all_frozen):
    """Now trivially true, because the canonical split itself is clean."""
    eligible = splits.training_eligible(list(all_frozen))
    leaked = [s for s in eligible if siblings.is_held_out(s.axes, s.template_id)]
    assert leaked == [], [s.scenario_id for s in leaked]
    assert len(eligible) == 15


def test_the_1_1_0_filter_would_have_kept_59_scenarios(all_frozen):
    """What the runtime filter achieved, kept as the record of why it was not enough.

    59 eligible scenarios under 1.0.0 partitions -- clean at the selector, while
    the split beneath it still shipped 85 counterfactuals as training data.
    """
    filtered = [
        s
        for s in all_frozen
        if splits.partition_v1_0_0(s) is splits.Partition.TRAIN
        and not siblings.is_held_out(s.axes, s.template_id)
    ]
    assert len(filtered) == 59


def test_every_eligible_scenario_is_in_the_train_partition(all_frozen):
    for scenario in splits.training_eligible(list(all_frozen)):
        assert splits.partition_of(scenario) is splits.Partition.TRAIN


def test_ineligibility_names_a_specific_reason(all_frozen):
    """Under 1.2.0 only one reason can fire, because the split is clean.

    ``REGISTERED_HELD_OUT`` is kept as a backstop against a split regression, so
    its absence here is the property under test, not a gap.
    """
    reasons = {
        splits.training_ineligibility(s)
        for s in all_frozen
        if not splits.is_training_eligible(s)
    }
    assert reasons == {splits.Ineligibility.NOT_TRAIN_PARTITION}


def test_the_split_versions_are_all_preserved(inventory):
    """1.0.0 and 1.1.0 stay readable; 1.2.0 is canonical."""
    assert splits.SPLIT_VERSION == "1.1.0"
    assert splits.ELIGIBILITY_VERSION == "1.1.0"
    assert splits.SPLIT_VERSION_CANONICAL == "1.2.0"
    assert inventory.version == "1.2.0"
    assert callable(splits.partition_v1_0_0)


# --------------------------------------------------------------------------
# the inventory
# --------------------------------------------------------------------------


def test_the_inventory_covers_every_family_and_partition(all_frozen, inventory):
    families = {s.family for s in all_frozen}
    seen = {key.split("|")[0] for key in inventory.counts}
    assert seen == families
    assert sum(inventory.counts.values()) == len(all_frozen)


def test_the_inventory_shows_no_holdout_reaching_canonical_training(inventory):
    """The inventory reads the canonical split, so this is now zero."""
    assert sum(inventory.held_out_in_train.values()) == 0
    assert inventory.held_out_in_train == {}
    assert len(inventory.held_out_values) == 11
    assert inventory.eligible == 15


def test_validation_and_evaluation_carry_the_holdouts(all_frozen):
    """They have to go somewhere; 1.2.0 sends them to the held-out partitions."""
    for partition in (splits.Partition.VALIDATION, splits.Partition.EVALUATION):
        members = splits.select(list(all_frozen), partition)
        held = [s for s in members if siblings.is_held_out(s.axes, s.template_id)]
        assert held, partition


# --------------------------------------------------------------------------
# sibling groups still follow the approved partition rules
# --------------------------------------------------------------------------


def test_a_counterfactual_and_its_sibling_share_a_partition(all_frozen):
    """Approved rule 1, unchanged by the eligibility layer."""
    by_key: dict[str, set[str]] = {}
    for scenario in all_frozen:
        by_key.setdefault(splits.pair_key(scenario), set()).add(
            splits.partition_of(scenario).value,
        )
    straddling = {k: v for k, v in by_key.items() if len(v) > 1}
    assert straddling == {}


def test_a_pairs_entities_never_straddle_the_split(all_frozen):
    """Approved rule 2: same seed and template implies the same partition."""
    by_identity: dict[tuple[str, int, str], set[str]] = {}
    for scenario in all_frozen:
        axes = scenario.axes
        if siblings.is_held_out(axes, scenario.template_id):
            axes = siblings.sibling_axes(axes, scenario.template_id)
        key = (scenario.template_id, scenario.root_seed, str(sorted(axes.items())))
        by_identity.setdefault(key, set()).add(splits.partition_of(scenario).value)
    assert all(len(v) == 1 for v in by_identity.values())


def test_the_1_1_0_layer_relabelled_nothing(all_frozen):
    """Provenance: 1.1.0 added a filter and moved no scenario.

    Asserted against ``partition_v1_0_0``, which 1.1.0 used unchanged. The
    partition moves belong to 1.2.0 and are audited in ``test_canonical_split``.
    """
    for scenario in all_frozen:
        bucket = int(splits.pair_key(scenario)[:8], 16) % 100
        expected = (
            splits.Partition.TRAIN
            if bucket < splits.TRAIN_SHARE
            else splits.Partition.VALIDATION
            if bucket < splits.TRAIN_SHARE + splits.VALIDATION_SHARE
            else splits.Partition.EVALUATION
        )
        assert splits.partition_v1_0_0(scenario) is expected
