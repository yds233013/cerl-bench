"""Registered holdouts versus actual training data.

The check that found the leak: a partition label says which side of a *paired
comparison* a group belongs to, and says nothing about whether a policy may be
trained on it. Those are different questions, and answering the second with the
first put 85 registered counterfactuals inside the training partition.
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


def test_the_train_partition_does_contain_registered_held_out_values(all_frozen):
    """Not a bug in the partition -- a consequence of its approved rule.

    Partition rule 1 puts a counterfactual and its ID sibling together on
    purpose. So held-out values appear in whichever partition their pair lands
    in, train included. Asserting it here keeps the eligibility layer's reason
    for existing visible rather than folkloric.
    """
    train = splits.select(list(all_frozen), splits.Partition.TRAIN)
    held_out = [s for s in train if siblings.is_held_out(s.axes, s.template_id)]
    assert len(held_out) == 85, len(held_out)
    assert len(train) == 144


def test_training_eligible_scenarios_contain_no_registered_held_out_value(all_frozen):
    """The assertion the closeout asked for. This is the leakage gate."""
    eligible = splits.training_eligible(list(all_frozen))
    leaked = [s for s in eligible if siblings.is_held_out(s.axes, s.template_id)]
    assert leaked == [], [s.scenario_id for s in leaked]
    assert len(eligible) == 59


def test_every_eligible_scenario_is_in_the_train_partition(all_frozen):
    for scenario in splits.training_eligible(list(all_frozen)):
        assert splits.partition_of(scenario) is splits.Partition.TRAIN


def test_ineligibility_names_a_specific_reason(all_frozen):
    """Two independent reasons; a scenario can be excluded for either."""
    reasons = {
        splits.training_ineligibility(s)
        for s in all_frozen
        if not splits.is_training_eligible(s)
    }
    assert reasons == {
        splits.Ineligibility.NOT_TRAIN_PARTITION,
        splits.Ineligibility.REGISTERED_HELD_OUT,
    }


def test_the_eligibility_layer_is_versioned(inventory):
    assert splits.ELIGIBILITY_VERSION == "1.1.0"
    assert splits.SPLIT_VERSION == splits.ELIGIBILITY_VERSION
    assert inventory.version == "1.1.0"


# --------------------------------------------------------------------------
# the inventory
# --------------------------------------------------------------------------


def test_the_inventory_covers_every_family_and_partition(all_frozen, inventory):
    families = {s.family for s in all_frozen}
    seen = {key.split("|")[0] for key in inventory.counts}
    assert seen == families
    assert sum(inventory.counts.values()) == len(all_frozen)


def test_the_inventory_records_which_held_out_values_reach_train(inventory):
    assert sum(inventory.held_out_in_train.values()) == 85
    assert len(inventory.held_out_values) == 11


def test_no_partition_is_free_of_held_out_values_by_accident(all_frozen):
    """Validation carries them too, so eligibility is not a train-only concern."""
    validation = splits.select(list(all_frozen), splits.Partition.VALIDATION)
    assert any(siblings.is_held_out(s.axes, s.template_id) for s in validation)


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


def test_no_scenario_moved_partition_in_the_1_1_0_correction(all_frozen):
    """Provenance: the fix added a layer, it did not relabel anything.

    Partition assignment is still a pure function of the pair key, so the 1.0.0
    partition of every scenario is recomputable and unchanged.
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
        assert splits.partition_of(scenario) is expected
