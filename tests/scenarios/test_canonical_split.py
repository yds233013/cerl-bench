"""Criterion 42 against the **canonical** split, not against a consumer's filter.

Split 1.1.0 added a runtime eligibility filter in the pilot selector. That kept
the pilot clean and left the split itself broken: 85 registered counterfactuals
sat in the training partition, and anything reading ``partition_of`` directly --
a future training loop included -- would have consumed them.

These tests therefore read ``splits.partition_of`` and the committed manifest.
None of them goes through the pilot selector. A filter in one consumer is not a
fixed split, and a test that only exercises the filter cannot tell the two apart.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from cerl.eval import splits
from cerl.scenario import siblings
from cerl.scenario.lexicon import COMPANY_SHARDS


@pytest.fixture(scope="module")
def manifest():
    return splits.manifest()


# --------------------------------------------------------------------------
# guarantee 1 -- no training scenario carries a registered held-out value
# --------------------------------------------------------------------------


def test_the_canonical_training_partition_contains_no_registered_holdout(all_frozen):
    """Criterion 42, second clause, read straight off the canonical split."""
    train = [s for s in all_frozen if splits.partition_of(s) is splits.Partition.TRAIN]
    leaked = [s for s in train if siblings.is_held_out(s.axes, s.template_id)]
    assert leaked == [], [s.scenario_id for s in leaked]
    assert train, "the training partition is empty; that is a different defect"


def test_no_training_scenario_carries_a_holdout_on_any_axis(all_frozen):
    """Checks every intervention axis, not just the one a pair is keyed on."""
    for scenario in all_frozen:
        if splits.partition_of(scenario) is not splits.Partition.TRAIN:
            continue
        pairing = siblings.pairing_for(scenario.template_id)
        for intervention in pairing.interventions:
            value = scenario.axes.get(intervention.axis, "")
            assert not intervention.is_held_out(value), (
                f"{scenario.scenario_id}: {intervention.axis}={value} is registered "
                f"held out for {scenario.family}"
            )


def test_the_1_1_0_eligibility_filter_is_now_redundant(all_frozen):
    """With the split fixed, the filter excludes nothing inside training.

    Its remaining job is as a backstop against a future split regression, which
    is why it is kept rather than deleted.
    """
    train = [s for s in all_frozen if splits.partition_of(s) is splits.Partition.TRAIN]
    eligible = splits.training_eligible(list(all_frozen))
    assert sorted(s.scenario_id for s in train) == sorted(
        s.scenario_id for s in eligible
    )


# --------------------------------------------------------------------------
# guarantee 2 -- lexicon shards
# --------------------------------------------------------------------------


def test_lexicon_shard_definitions_are_pairwise_disjoint():
    """The shard *pools* are disjoint. Necessary, and not sufficient -- see below."""
    shards = {name: set(values) for name, values in COMPANY_SHARDS.items()}
    names = sorted(shards)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not shards[left] & shards[right], (left, right)


def test_the_corpus_does_not_yet_use_disjoint_shards_across_partitions(all_frozen):
    """**Criterion 42, first clause, FAILS.** Asserted so it cannot be forgotten.

    Every frozen scenario draws from the ``core`` shard, so partitions share
    entity names. The shard machinery exists and its pools are disjoint; the
    corpus was simply generated before it was wired in.

    This cannot be repaired by moving groups between partitions -- the names are
    baked into the frozen files, and every partition would still be drawing from
    ``core``. Fixing it means regenerating the corpus against per-partition
    shards, which changes every scenario hash and needs a generator version bump.

    The test asserts the *current* state deliberately. When the corpus is
    regenerated this test must be inverted, and that is the intended signal.
    """
    by_partition: dict[str, set[str]] = {}
    for scenario in all_frozen:
        key = splits.partition_of(scenario).value
        by_partition.setdefault(key, set()).update(
            customer.display_name
            for customer in scenario.world.billing.customers.values()
        )
    partitions = sorted(by_partition)
    overlaps = {
        (left, right): by_partition[left] & by_partition[right]
        for i, left in enumerate(partitions)
        for right in partitions[i + 1 :]
    }
    assert all(overlaps.values()), (
        "partitions no longer share entity names -- the corpus was regenerated "
        "against per-partition shards. Invert this test and mark Criterion 42's "
        "first clause PASS."
    )


# --------------------------------------------------------------------------
# guarantee 3 -- sibling pairs stay together
# --------------------------------------------------------------------------


def test_every_sibling_group_is_in_exactly_one_partition(all_frozen):
    by_key: dict[str, set[str]] = {}
    for scenario in all_frozen:
        by_key.setdefault(splits.pair_key(scenario), set()).add(
            splits.partition_of(scenario).value,
        )
    straddling = {k: v for k, v in by_key.items() if len(v) > 1}
    assert straddling == {}


def test_a_counterfactual_and_its_id_sibling_share_a_partition(all_frozen):
    """Stated on the scenarios rather than on the grouping key."""
    by_id = {s.scenario_id: s for s in all_frozen}
    checked = 0
    for scenario in all_frozen:
        if not siblings.is_held_out(scenario.axes, scenario.template_id):
            continue
        sibling_axes = siblings.sibling_axes(scenario.axes, scenario.template_id)
        for candidate in by_id.values():
            if (
                candidate.template_id == scenario.template_id
                and candidate.root_seed == scenario.root_seed
                and candidate.axes == sibling_axes
            ):
                assert splits.partition_of(candidate) is splits.partition_of(scenario)
                checked += 1
    assert checked > 0, "no CF/ID pairs were actually compared"


# --------------------------------------------------------------------------
# guarantee 4 -- nothing omitted, relabelled, or duplicated
# --------------------------------------------------------------------------


def test_the_manifest_covers_the_corpus_exactly(all_frozen, manifest):
    problems = splits.verify_manifest_totality(list(all_frozen), manifest)
    assert problems == ()
    assert len(manifest.scenario_ids()) == len(all_frozen) == 190


def test_no_scenario_appears_in_two_groups(manifest):
    seen: set[str] = set()
    for group in manifest.groups:
        for member in group.members:
            assert member not in seen, member
            seen.add(member)


def test_the_committed_manifest_matches_a_fresh_build(all_frozen, manifest):
    """The file on disk is the artifact; drift from the rule is a defect."""
    rebuilt = splits.build_manifest(list(all_frozen))
    assert rebuilt == manifest


def test_the_manifest_file_is_hash_pinned():
    from cerl.core import content_hash

    payload = json.loads(splits.MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_hash")
    assert recorded == content_hash(payload)


# --------------------------------------------------------------------------
# guarantee 5 -- holdouts are namespaced by family and axis
# --------------------------------------------------------------------------


def test_held_out_values_are_namespaced_by_family_and_axis(manifest):
    """``signal_count=1`` and ``approval=expired`` are unrelated facts.

    A bare value would collide the moment two families shared an axis name --
    and W1 and W2 both have an approval-shaped axis with values ``expired`` and
    ``missing_unobtainable``.
    """
    for group in manifest.groups:
        if not group.moved:
            continue
        assert "held-out" in group.reason
        for label in group.reason.split("held-out ", 1)[1].split(", "):
            assert "/" in label and "=" in label, label
            family, rest = label.split("/", 1)
            axis, _, value = rest.partition("=")
            assert family and axis and value, label


def test_the_two_approval_axes_are_not_conflated():
    w1 = siblings.pairing_for("dup_profile_reconciliation").interventions[0]
    w2 = siblings.pairing_for("dup_charge_threshold").interventions[0]
    assert w1.axis == "merge_approval"
    assert w2.axis == "approval"
    shared = set(w1.held_out) & set(w2.held_out)
    assert shared, "expected the families to share value names, hence the namespace"


# --------------------------------------------------------------------------
# migration provenance
# --------------------------------------------------------------------------


def test_only_cf_bearing_training_groups_moved(manifest):
    for group in manifest.groups:
        if group.moved:
            assert group.cf_bearing, group.pair_key
            assert group.partition_v1_0_0 is splits.Partition.TRAIN
            assert group.partition is not splits.Partition.TRAIN
        elif group.cf_bearing:
            assert group.partition_v1_0_0 is not splits.Partition.TRAIN


def test_pure_id_groups_keep_their_1_0_0_assignment(all_frozen, manifest):
    for group in manifest.groups:
        if not group.cf_bearing:
            assert group.partition is group.partition_v1_0_0, group.pair_key


def test_the_1_0_0_rule_is_preserved_verbatim(all_frozen):
    """1.0.0 stays computable, so the migration is auditable in both directions."""
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


def test_no_scenario_file_changed_in_the_migration():
    """Only partition labels moved.

    Checked against the corpus manifest rather than by assertion: every frozen
    file's committed sha256 must still match the bytes on disk. If the migration
    had regenerated anything, this fails.
    """
    from cerl.core import hash_text

    corpus = json.loads(pathlib.Path("scenarios/manifest.json").read_text())
    assert corpus["count"] == 190
    for entry in corpus["scenarios"]:
        path = pathlib.Path("scenarios/frozen") / pathlib.Path(entry["file"]).name
        actual = hash_text(path.read_text(encoding="utf-8"))
        assert actual == entry["sha256"], entry["scenario_id"]
        assert entry["generator_version"]
        assert isinstance(entry["root_seed"], int)


def test_the_split_manifest_does_not_duplicate_scenario_provenance():
    """Provenance stays in one place; the split manifest only carries labels.

    Two copies of a hash is two things to keep in sync, and the split has no
    business asserting what a scenario contains.
    """
    payload = json.loads(splits.MANIFEST_PATH.read_text(encoding="utf-8"))
    blob = json.dumps(payload)
    assert "sha256" not in blob
    assert "generator_version" not in blob
