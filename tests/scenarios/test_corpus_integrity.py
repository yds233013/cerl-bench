"""Corpus 2.0.0 read off the committed files.

Everything here inspects `scenarios/v2/` on disk. Regeneration is only correct if
the artifacts it produced are correct, and a test that re-derives its expectation
from the generator would agree with a broken generator.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cerl.core import hash_text
from cerl.scenario import corpus, freeze, lexicon
from tests.helpers import FROZEN_DIR, frozen_paths

EXPECTED_COUNT = 190


@pytest.fixture(scope="module")
def manifest():
    return json.loads(corpus.CANONICAL.manifest.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# size, identity, and the ways a corpus silently shrinks
# --------------------------------------------------------------------------


def test_the_corpus_has_the_expected_size(all_frozen, manifest):
    assert len(all_frozen) == EXPECTED_COUNT
    assert len(frozen_paths()) == EXPECTED_COUNT
    assert manifest["count"] == EXPECTED_COUNT
    assert len(manifest["scenarios"]) == EXPECTED_COUNT


def test_no_duplicate_scenario_ids(all_frozen):
    """A colliding id overwrites a file and the corpus shrinks in silence."""
    ids = [s.scenario_id for s in all_frozen]
    duplicates = [sid for sid, n in Counter(ids).items() if n > 1]
    assert duplicates == []


def test_no_duplicate_filenames(manifest):
    """One level below an id collision, with the same consequence."""
    files = [entry["file"] for entry in manifest["scenarios"]]
    duplicates = [name for name, n in Counter(files).items() if n > 1]
    assert duplicates == []


def test_every_planned_instance_was_written(all_frozen):
    """The plan and the corpus agree, so nothing was lost between them."""
    from cerl.scenario.families import registry

    planned = sum(
        len(registry.plan_for(template_id)())
        for template_id in registry.template_ids()
    )
    assert planned == len(all_frozen) == EXPECTED_COUNT


def test_scenario_hashes_match_the_manifest(manifest):
    for entry in manifest["scenarios"]:
        path = FROZEN_DIR / Path(entry["file"]).name
        assert path.exists(), entry["scenario_id"]
        assert hash_text(path.read_text(encoding="utf-8")) == entry["sha256"], (
            entry["scenario_id"]
        )


def test_the_manifest_records_the_full_provenance(manifest):
    assert manifest["corpus_version"] == corpus.CORPUS_VERSION == "2.0.0"
    assert manifest["generator_version"] == "w2-2.0.0"
    assert manifest["shard_version"] == lexicon.SHARD_VERSION == "2.0.0"
    for entry in manifest["scenarios"]:
        assert entry["generator_version"] == "w2-2.0.0"
        assert isinstance(entry["root_seed"], int)
        assert entry["sha256"]


# --------------------------------------------------------------------------
# byte stability
# --------------------------------------------------------------------------


def test_re_materializing_reproduces_each_file_byte_exactly(all_frozen):
    """Regeneration is only reproducible if it is *re*-producible."""
    for scenario in all_frozen:
        again = freeze.materialize(
            scenario.template_id,
            scenario.axes,
            scenario.root_seed,
            freeze.shard_for(
                scenario.template_id, scenario.axes, scenario.root_seed,
            ),
        )
        again = again.model_copy(
            update={"oracle_tool_calls": scenario.oracle_tool_calls},
        )
        assert freeze.to_json(again) == freeze.to_json(scenario), scenario.scenario_id


def test_the_shard_resolver_agrees_with_every_committed_file(all_frozen):
    """One answer to "which shard", used by the generator and by tests alike."""
    for scenario in all_frozen:
        resolved = freeze.shard_for(
            scenario.template_id, scenario.axes, scenario.root_seed,
        )
        assert resolved == scenario.lexicon_shard, scenario.scenario_id


# --------------------------------------------------------------------------
# what the regeneration must not have changed
# --------------------------------------------------------------------------


def test_the_w1_identity_evidence_challenge_set_survives(all_frozen):
    """Preserved through the regeneration, and still a separate challenge set.

    It is not a matched-pair set and is not counted as one -- see
    ``docs/w1-scope.md``. What must not happen is it quietly disappearing in a
    corpus rebuild.
    """
    w1 = [s for s in all_frozen if s.template_id == "dup_profile_reconciliation"]
    values = {s.axes["identity_evidence"] for s in w1}
    assert values == {
        "strong_match",
        "conflicting_external_ref",
        "name_only_similarity",
    }
    assert len(w1) == 36


def test_the_identity_evidence_difficulty_spread_is_unchanged(all_frozen):
    """The measured 12/10/9 is a property of the workflows, not of the names."""
    baseline = {
        s.axes["identity_evidence"]: s.oracle_tool_calls
        for s in all_frozen
        if s.template_id == "dup_profile_reconciliation"
        and s.axes["merge_approval"] == "valid"
        and s.axes["dispute_state"] == "none"
        and s.axes["tool_reliability"] == "stable"
        and s.axes["prior_progress"] == "none"
    }
    assert baseline == {
        "strong_match": 12,
        "name_only_similarity": 10,
        "conflicting_external_ref": 9,
    }, baseline


def test_entity_cardinality_is_matched_within_every_declared_pair(all_frozen):
    """Criterion 41's structural half, over the regenerated files."""
    from cerl.scenario import siblings

    by_id = {s.scenario_id: s for s in all_frozen}
    compared = 0
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
                assert len(candidate.world.billing.customers) == len(
                    scenario.world.billing.customers,
                ), scenario.scenario_id
                assert len(candidate.world.billing.charges) == len(
                    scenario.world.billing.charges,
                ), scenario.scenario_id
                compared += 1
    assert compared > 0


def test_brief_lengths_stay_within_fifteen_percent_within_a_pair(all_frozen):
    from cerl.scenario import siblings

    by_id = {s.scenario_id: s for s in all_frozen}
    compared = 0
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
                longer = max(len(candidate.brief), len(scenario.brief))
                shorter = min(len(candidate.brief), len(scenario.brief))
                assert longer <= shorter * 1.15, scenario.scenario_id
                compared += 1
    assert compared > 0


def test_all_data_is_still_synthetic(all_frozen):
    """New names, same rule: nothing real."""
    pooled = set()
    for shard in lexicon.shard_names():
        pooled |= lexicon.all_values(shard)
    for scenario in all_frozen:
        for customer in scenario.world.billing.customers.values():
            assert customer.email.endswith(".example"), customer.email
        for user in scenario.world.slack.users.values():
            assert user.display_name in pooled or user.handle in pooled


def test_a_partial_freeze_cannot_overwrite_the_committed_split(tmp_path):
    """Regression: it could, and it did.

    ``--split-out`` briefly defaulted to the canonical path, so a six-scenario
    test freeze replaced the corpus's 86-group split manifest with an 11-group
    one. The default now follows ``--manifest``, so a freeze writing elsewhere
    writes its split elsewhere too.
    """
    from typer.testing import CliRunner

    from cerl.cli import app

    before = corpus.CANONICAL.split_manifest.read_text(encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "freeze", "--family", "duplicate_charge_approval", "--limit", "4",
            "--out", str(tmp_path / "frozen"),
            "--gold-out", str(tmp_path / "gold"),
            "--manifest", str(tmp_path / "manifest.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "split_manifest.json").exists()
    assert corpus.CANONICAL.split_manifest.read_text(encoding="utf-8") == before
