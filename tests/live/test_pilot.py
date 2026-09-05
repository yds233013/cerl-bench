"""The pilot: selection, split integrity, projection, execution, regeneration."""

from __future__ import annotations

import pytest

from cerl.agents.budget import MockTokenCounter, SpendLedger
from cerl.agents.synthetic_transport import (
    SyntheticFailure,
    SyntheticTransport,
    text_turn,
    tool_turn,
)
from cerl.eval import manifest as manifest_module
from cerl.eval import pilot, splits, verify_run
from cerl.reference.registry import families as oracle_families


@pytest.fixture(scope="module")
def projection(all_frozen):
    return pilot.project(list(all_frozen))


@pytest.fixture(scope="module")
def chosen(all_frozen):
    return list(pilot.select(list(all_frozen)))


# --------------------------------------------------------------------------
# selection and split integrity
# --------------------------------------------------------------------------


def test_the_selection_draws_only_from_the_train_partition(chosen):
    """A pilot whose outcomes inform fixes must not consume held-out scenarios."""
    assert chosen
    for scenario in chosen:
        assert splits.partition_of(scenario) is splits.Partition.TRAIN, scenario.scenario_id


def test_the_audit_reports_a_clean_split(all_frozen, projection):
    report = projection.audit
    assert report.partition == "train"
    assert report.held_out_partitions_touched == ()
    assert report.uncoverable_branches == ()
    assert report.clean
    assert dict(report.partitions_touched) == {"train": len(projection.episodes)}


def test_every_branch_of_every_family_is_covered(all_frozen, projection):
    corpus = {f"{s.family}/{s.branch}" for s in all_frozen}
    assert set(projection.audit.branch_coverage) == corpus
    assert len(corpus) == 10
    assert {k.split("/")[0] for k in corpus} == set(oracle_families())
    assert set(projection.audit.branch_coverage.values()) == {pilot.EPISODES_PER_BRANCH}


def test_the_audit_surfaces_a_coverage_conflict_rather_than_hiding_it(all_frozen):
    """The evaluation partition cannot supply every branch. Say so, don't fix it.

    Silently reaching into another partition to complete coverage is exactly the
    move that destroys a held-out split, so the audit reports the conflict and
    leaves the decision to a person.
    """
    report = pilot.audit(list(all_frozen), splits.Partition.EVALUATION)
    assert report.uncoverable_branches, "expected evaluation to be short of branches"
    assert not report.clean
    assert report.held_out_partitions_touched == ()


def test_selection_is_deterministic(all_frozen):
    forward = pilot.select(list(all_frozen))
    backward = pilot.select(list(reversed(list(all_frozen))))
    assert [s.scenario_id for s in forward] == [s.scenario_id for s in backward]


def test_sibling_groups_are_recorded_so_leakage_is_checkable(projection):
    assert projection.audit.sibling_groups
    assert len(set(projection.audit.sibling_groups)) == len(projection.audit.sibling_groups)


def test_the_rest_of_the_corpus_is_not_relabelled(all_frozen, chosen):
    """Unselected scenarios keep their partitions; nothing is moved."""
    ids = {s.scenario_id for s in chosen}
    rest = [s for s in all_frozen if s.scenario_id not in ids]
    assert len(rest) == len(all_frozen) - len(chosen)
    partitions = {splits.partition_of(s).value for s in rest}
    assert {"validation", "evaluation"} <= partitions


# --------------------------------------------------------------------------
# projection: an estimate, labelled as one
# --------------------------------------------------------------------------


def test_the_projection_is_labelled_an_estimate(projection):
    assert projection.counter_name in {"heuristic", "mock", "provider-count_tokens"}
    assert projection.model == "claude-opus-5"
    assert projection.max_tokens == 2048


def test_worst_case_far_exceeds_the_expected_case(projection):
    """The gap is the finding: a mean is not a spending bound."""
    expected = projection.total_cents(worst_case=False, cached=True)
    worst = projection.total_cents(worst_case=True, cached=True)
    assert worst > expected * 5


def test_the_recommended_cap_exceeds_the_modelled_worst_case(projection):
    cap = projection.recommended_cap_cents()
    assert cap > projection.total_cents(worst_case=True, cached=True)
    assert cap % 100 == 0


def test_caching_lowers_the_projection(projection):
    assert projection.total_cents(worst_case=True, cached=True) < projection.total_cents(
        worst_case=True, cached=False,
    )


def test_the_published_proposal_figures_still_hold(projection):
    assert len(projection.episodes) == 20
    expected = projection.total_cents(worst_case=False, cached=True)
    worst = projection.total_cents(worst_case=True, cached=True)
    assert expected == pytest.approx(418.0, abs=25.0), expected
    assert worst == pytest.approx(5054.0, abs=200.0), worst
    assert projection.recommended_cap_cents() == 5100


# --------------------------------------------------------------------------
# execution against the synthetic transport
# --------------------------------------------------------------------------


def _synthetic_run(scenarios, steps=None, cap=100_000.0, ledger_path=None):
    ledger = SpendLedger(cap_cents=cap, model=pilot.PILOT_MODEL, counter_name="mock")
    transport = SyntheticTransport(
        steps or [], default=text_turn("no scripted turn"),
    )
    client = pilot.build_client(
        ledger, transport=transport, counter=MockTokenCounter(),
    )
    return pilot.execute(
        scenarios, client, ledger, source=transport.source, ledger_path=ledger_path,
    )


def test_a_synthetic_run_completes_and_is_labelled_synthetic(chosen):
    result = _synthetic_run(chosen[:2])
    assert result.completed == 2
    assert result.source == "synthetic"
    assert result.manifest.verification_mode == "synthetic"
    assert result.manifest.agent.provider == "synthetic-transport"
    assert all(t.source == "synthetic" for t in result.transcripts)


def test_a_synthetic_run_is_never_recorded_as_a_model_result(chosen):
    """The strongest guard: no artifact from this path may read as live."""
    result = _synthetic_run(chosen[:2])
    assert result.manifest.verification_mode != "live"
    for transcript in result.transcripts:
        for entry in transcript.entries:
            assert entry.response.source == "synthetic"


def test_tool_calls_drive_real_actions(chosen):
    scenario = chosen[0]
    ticket = str(scenario.variables["ticket"])
    result = _synthetic_run(
        [scenario],
        steps=[
            tool_turn("tickets__get", {"ticket_id": ticket}),
            tool_turn("abstain", {"reason": "scripted"}),
        ],
    )
    actions = result.manifest.episodes[0].actions
    assert [a.kind for a in actions][:2] == ["tickets.get", "abstain"]


def test_a_malformed_turn_is_scored_rather_than_swallowed(chosen):
    result = _synthetic_run([chosen[0]], steps=[text_turn("I would refund it.")])
    kinds = [a.kind for a in result.manifest.episodes[0].actions]
    assert "malformed" in kinds[0]


def test_an_interrupted_run_keeps_the_episodes_already_paid_for(chosen):
    """Discarding completed episodes would waste money already spent."""
    ledger = SpendLedger(cap_cents=100_000.0, model=pilot.PILOT_MODEL)
    # The first episode terminates immediately. The second exhausts its retries
    # (max_retries=2, so three consecutive failures), which aborts the run.
    steps = [
        tool_turn("abstain", {"reason": "done"}),
        SyntheticFailure("boom"),
        SyntheticFailure("boom"),
        SyntheticFailure("boom"),
    ]
    transport = SyntheticTransport(steps)
    client = pilot.build_client(ledger, transport=transport, counter=MockTokenCounter())
    result = pilot.execute(
        chosen[:3], client, ledger, source="synthetic",
    )
    assert result.completed >= 1
    assert result.interrupted
    assert len(result.manifest.episodes) == result.completed


def test_the_budget_stops_a_run_and_the_partial_result_is_kept(chosen):
    result = _synthetic_run(chosen, cap=3.0)
    assert result.interrupted
    assert result.spend["cap_cents"] == 3.0
    assert float(result.spend["confirmed_cents"]) <= 3.0


def test_spend_is_recorded_in_the_manifest_metrics(chosen):
    result = _synthetic_run(chosen[:2])
    spend = dict(result.manifest.metrics)["spend"]
    assert {"confirmed_cents", "reserved_cents", "unresolved_cents"} <= set(spend)


def test_the_ledger_persists_across_the_run(chosen, tmp_path):
    path = tmp_path / "ledger.json"
    _synthetic_run(chosen[:2], ledger_path=path)
    assert path.exists()
    resumed = SpendLedger.resume(path, 100_000.0, pilot.PILOT_MODEL)
    assert resumed.confirmed_cents > 0.0


# --------------------------------------------------------------------------
# regeneration and offline verification
# --------------------------------------------------------------------------


def test_a_recorded_run_regenerates_byte_exactly_from_its_cache(chosen):
    result = _synthetic_run(chosen[:3])
    report = pilot.regenerate(chosen[:3], result.manifest, list(result.transcripts))
    assert report.ok, report.summary()
    assert report.matched == 3


def test_a_cache_miss_is_reported_rather_than_falling_back_to_a_live_call(chosen):
    result = _synthetic_run(chosen[:2])
    gutted = [t.model_copy(update={"entries": ()}) for t in result.transcripts]
    report = pilot.regenerate(chosen[:2], result.manifest, gutted)
    assert not report.ok
    assert report.cache_misses
    assert "CACHE MISS" in report.summary()


def test_a_recorded_run_verifies_offline_by_action_replay(chosen, tmp_path):
    result = _synthetic_run(chosen[:3])
    path = manifest_module.write(result.manifest, tmp_path / "run.json")
    reloaded = manifest_module.load(path)
    from cerl.scenario import freeze as freeze_module

    report = verify_run.verify_manifest(reloaded, freeze_module.FROZEN_DIR)
    assert report.ok, report.summary()


def test_transcripts_round_trip_through_disk(chosen, tmp_path):
    result = _synthetic_run(chosen[:2])
    path = pilot.write_transcripts(result.transcripts, tmp_path / "t.json")
    assert pilot.read_transcripts(path) == result.transcripts
