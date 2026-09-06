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


def test_the_corpus_uses_disjoint_shards_across_partitions(all_frozen):
    """**Criterion 42, first clause, PASSES on corpus 2.0.0.**

    The previous version of this test asserted the opposite and said so: corpus
    1.x drew every scenario from ``core``, and it could not be repaired by
    relabelling because the names were baked into the files. It was inverted
    when the corpus was regenerated, which is exactly the signal it was written
    to give.

    Checked over the committed files, on the values an agent can actually read,
    rather than over the pool definitions -- disjoint pools are what 1.x already
    had.
    """
    by_partition: dict[str, set[tuple[str, str]]] = {}
    for scenario in all_frozen:
        key = splits.partition_of(scenario).value
        bucket = by_partition.setdefault(key, set())
        for customer in scenario.world.billing.customers.values():
            bucket.add(("customer", customer.display_name))
            bucket.add(("email", customer.email))
        for user in scenario.world.slack.users.values():
            bucket.add(("handle", user.handle))
            bucket.add(("user", user.display_name))

    partitions_seen = sorted(by_partition)
    assert partitions_seen == ["evaluation", "train", "validation"]
    for i, left in enumerate(partitions_seen):
        for right in partitions_seen[i + 1 :]:
            overlap = by_partition[left] & by_partition[right]
            assert overlap == set(), (left, right, sorted(overlap)[:5])


def test_every_scenario_records_the_shard_it_was_built_from(all_frozen):
    """Provenance in the file, not only in the manifest.

    A scenario that ended up in the wrong partition is then detectable from its
    own contents, without trusting the manifest that placed it.
    """
    for scenario in all_frozen:
        assert scenario.corpus_version == "2.0.0", scenario.scenario_id
        assert scenario.lexicon_shard == scenario.partition
        assert scenario.partition == splits.partition_of(scenario).value


def test_pool_definitions_and_corpus_contents_agree(all_frozen):
    """Every lexicon value in a file belongs to that partition's pool.

    The other direction from the disjointness test: not merely that partitions
    do not overlap, but that each draws from the pool it is supposed to.
    """
    from cerl.scenario import lexicon

    for scenario in all_frozen:
        allowed = lexicon.all_values(scenario.lexicon_shard)
        foreign = {
            other
            for other in lexicon.shard_names()
            if other != scenario.lexicon_shard
        }
        text = json.dumps(scenario.model_dump(mode="json"), sort_keys=True)
        for shard in foreign:
            for value in lexicon.all_values(shard) - allowed:
                assert value not in text, (scenario.scenario_id, shard, value)


def test_the_legacy_corpus_is_untouched():
    """Corpus 1.x stays exactly as it was: an immutable historical artifact."""
    from cerl.core import hash_text
    from tests.helpers import LEGACY_FROZEN_DIR, LEGACY_MANIFEST

    assert LEGACY_FROZEN_DIR.exists()
    legacy = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    assert legacy["count"] == 190
    assert legacy["generator_version"] == "w2-1.0.0"
    for entry in legacy["scenarios"]:
        path = LEGACY_FROZEN_DIR / pathlib.Path(entry["file"]).name
        assert hash_text(path.read_text(encoding="utf-8")) == entry["sha256"]


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
        assert "holdout" in group.reason
        assert "(" in group.reason and group.reason.endswith(")")
        for label in group.reason.rsplit("(", 1)[1].rstrip(")").split(", "):
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


def test_every_scenario_hash_matches_the_manifest():
    """The committed sha256 must match the bytes on disk, file by file."""
    from cerl.core import hash_text

    corpus = json.loads(pathlib.Path("scenarios/v2/manifest.json").read_text())
    assert corpus["count"] == 190
    assert corpus["corpus_version"] == "2.0.0"
    assert corpus["generator_version"] == "w2-2.0.0"
    assert corpus["shard_version"] == "2.0.0"
    for entry in corpus["scenarios"]:
        path = pathlib.Path("scenarios/v2/frozen") / pathlib.Path(entry["file"]).name
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
