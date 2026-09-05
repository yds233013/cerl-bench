"""The evaluation harness, its manifest, and offline verification.

The tests that matter most here are the negative ones. A verifier that only
confirms a good manifest proves nothing: it must *detect* an altered action, a
changed scenario, a doctored metric and an incompatible grading version, or it
is a rubber stamp.
"""

from __future__ import annotations

import pytest

from cerl.actions import BillingGetCharge
from cerl.core import ChargeId, FrozenMap
from cerl.eval import manifest as manifest_module
from cerl.eval import splits
from cerl.eval.metrics import (
    attempted_violation_rate,
    committed_violation_rate,
    decision_matrix,
)
from cerl.eval.runner import run_reference_evaluation
from cerl.eval.verify_run import verify_manifest
from cerl.scenario import siblings
from tests.helpers import FROZEN_DIR


@pytest.fixture(scope="module")
def small_run(all_frozen):
    """A manifest over a handful of scenarios from every family."""
    picked = []
    for family in sorted({s.family for s in all_frozen}):
        picked += [s for s in all_frozen if s.family == family][:3]
    # Deliberately include a scenario that fires a responder, so the
    # responder-tampering test has something to tamper with rather than skipping.
    with_responder = next(
        s for s in all_frozen if s.responders and "missing_obtainable" in s.scenario_id
    )
    if with_responder not in picked:
        picked.append(with_responder)
    return picked, run_reference_evaluation(picked, "oracle")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_every_partition_is_populated(all_frozen):
    summary = splits.summarize(all_frozen)
    assert summary.total() == len(all_frozen)
    for name in ("train", "validation", "evaluation"):
        assert summary.counts[name] > 0, name


def test_a_pair_never_straddles_the_split(all_frozen):
    """Splitting a CF/ID pair would put half a comparison in training."""
    by_key = {
        (s.template_id, tuple(sorted(s.axes.items())), s.root_seed): s for s in all_frozen
    }
    checked = 0
    for scenario in all_frozen:
        if not siblings.is_held_out(scenario.axes, scenario.template_id):
            continue
        sibling_axes = siblings.sibling_axes(scenario.axes, scenario.template_id)
        sibling = by_key[
            (scenario.template_id, tuple(sorted(sibling_axes.items())), scenario.root_seed)
        ]
        assert splits.partition_of(scenario) is splits.partition_of(sibling)
        checked += 1
    assert checked > 50


def test_shared_entities_within_a_pair_do_not_leak_across_the_split(all_frozen):
    """Both members of a pair share a seed and therefore the same entity names."""
    by_key = {
        (s.template_id, tuple(sorted(s.axes.items())), s.root_seed): s for s in all_frozen
    }
    for scenario in all_frozen:
        if not siblings.is_held_out(scenario.axes, scenario.template_id):
            continue
        sibling_axes = siblings.sibling_axes(scenario.axes, scenario.template_id)
        sibling = by_key[
            (scenario.template_id, tuple(sorted(sibling_axes.items())), scenario.root_seed)
        ]
        shared = set(scenario.world.billing.customers) & set(sibling.world.billing.customers)
        if shared:
            assert splits.partition_of(scenario) is splits.partition_of(sibling)


def test_partition_assignment_is_deterministic(all_frozen):
    first = [splits.partition_of(s) for s in all_frozen]
    second = [splits.partition_of(s) for s in all_frozen]
    assert first == second


# --------------------------------------------------------------------------
# what the manifest records
# --------------------------------------------------------------------------


def test_manifest_records_everything_required(small_run):
    _, manifest = small_run
    assert manifest.episodes
    assert manifest.agent.privilege_mode == "privileged"  # a reference policy
    assert manifest.agent.kind == "reference"
    assert manifest.predicate_library_hash
    assert manifest.verifier_version
    assert manifest.renderer_version
    assert manifest.split_version
    assert manifest.verification_mode == "replay"

    record = manifest.episodes[0]
    assert record.scenario_hash and record.initial_state_hash
    assert record.terminal_state_hash and record.trace_head_hash
    assert record.actions and record.steps
    assert record.partition in {"train", "validation", "evaluation"}
    assert record.required_decision
    assert record.tool_calls > 0
    # Two violation series, recorded separately.
    assert hasattr(record, "committed_violations")
    assert hasattr(record, "attempted_violations")


def test_every_step_records_time_hashes_and_responder_attribution(small_run):
    _, manifest = small_run
    for record in manifest.episodes:
        for step in record.steps:
            assert step.state_hash_before and step.state_hash_after
            assert step.entry_hash
            assert step.outcome
            assert int(step.logical_time) >= 0
            if str(step.origin) == "responder":
                assert step.responder_rule
                assert step.action is None
            else:
                assert step.action is not None


def test_metrics_keep_the_two_violation_series_apart(small_run):
    _, manifest = small_run
    metrics = dict(manifest.metrics)
    assert "committed_violation_rate" in metrics
    assert "attempted_violation_rate" in metrics
    assert "committed_by_class" in metrics
    assert "attempted_by_class" in metrics
    assert not any("unsafe_rate" in key for key in metrics)


def test_committed_and_attempted_rates_have_distinct_types(small_run):
    _, manifest = small_run
    episodes = list(manifest.episodes)
    committed = committed_violation_rate(episodes)
    attempted = attempted_violation_rate(episodes)
    assert isinstance(committed, float)
    assert isinstance(attempted, float)
    matrix = decision_matrix(episodes)
    assert matrix.matrix


# --------------------------------------------------------------------------
# offline verification: the positive case
# --------------------------------------------------------------------------


def test_a_genuine_manifest_verifies_offline(small_run, tmp_path):
    _, manifest = small_run
    path = manifest_module.write(manifest, tmp_path / "run.json")
    reloaded = manifest_module.load(path)
    report = verify_manifest(reloaded, FROZEN_DIR)
    assert report.ok, report.summary()
    assert report.episodes_checked == len(manifest.episodes)


def test_verification_needs_no_network_or_model(small_run, tmp_path, monkeypatch):
    """Assert the claim rather than assume it: break the socket layer first."""
    import socket

    def no_network(*_args, **_kwargs):
        raise AssertionError("verification attempted a network connection")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    _, manifest = small_run
    path = manifest_module.write(manifest, tmp_path / "run.json")
    report = verify_manifest(manifest_module.load(path), FROZEN_DIR)
    assert report.ok, report.summary()


# --------------------------------------------------------------------------
# offline verification: the negative cases that make it worth anything
# --------------------------------------------------------------------------


def test_an_altered_action_is_detected(small_run, tmp_path):
    _, manifest = small_run
    record = manifest.episodes[0]
    tampered_actions = (
        BillingGetCharge(charge_id=ChargeId.mint(999999)),
        *record.actions,
    )
    tampered = manifest.model_copy(
        update={
            "episodes": (
                record.model_copy(update={"actions": tampered_actions}),
                *manifest.episodes[1:],
            ),
        },
    )
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any("hash" in m.field for m in report.mismatches)


def test_a_doctored_verdict_is_detected(small_run):
    _, manifest = small_run
    record = manifest.episodes[0]
    lie = record.model_copy(
        update={"rubric": FrozenMap(dict.fromkeys(record.rubric, True)), "tool_calls": 1},
    )
    tampered = manifest.model_copy(update={"episodes": (lie, *manifest.episodes[1:])})
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any(m.field == "tool_calls" for m in report.mismatches)


def test_a_doctored_metric_block_is_detected(small_run):
    """Aggregates must follow from the episodes, not be asserted beside them."""
    _, manifest = small_run
    inflated = dict(manifest.metrics)
    inflated["safe_completion_rate"] = 0.0
    tampered = manifest.model_copy(update={"metrics": FrozenMap(inflated)})
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any(m.field == "metrics.safe_completion_rate" for m in report.mismatches)


def test_a_changed_scenario_is_detected(small_run):
    _, manifest = small_run
    record = manifest.episodes[0]
    tampered = manifest.model_copy(
        update={
            "episodes": (
                record.model_copy(update={"scenario_hash": "0" * 64}),
                *manifest.episodes[1:],
            ),
        },
    )
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any(m.field == "scenario_hash" for m in report.mismatches)


def test_a_changed_initial_state_is_detected(small_run):
    _, manifest = small_run
    record = manifest.episodes[0]
    tampered = manifest.model_copy(
        update={
            "episodes": (
                record.model_copy(update={"initial_state_hash": "0" * 64}),
                *manifest.episodes[1:],
            ),
        },
    )
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any(m.field == "initial_state_hash" for m in report.mismatches)


def test_a_changed_responder_transition_is_detected(small_run):
    _, manifest = small_run
    with_responder = next(
        (r for r in manifest.episodes if r.responder_transitions > 0), None,
    )
    assert with_responder is not None, (
        "the fixture must include a responder transition; a skip here would hide "
        "an unverified tamper path"
    )
    steps = list(with_responder.steps)
    index = next(i for i, s in enumerate(steps) if s.responder_rule)
    steps[index] = steps[index].model_copy(update={"responder_rule": "r_invented"})
    lie = with_responder.model_copy(update={"steps": tuple(steps)})
    tampered = manifest.model_copy(
        update={
            "episodes": tuple(
                lie if r.scenario_id == lie.scenario_id else r for r in manifest.episodes
            ),
        },
    )
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any("responder_rule" in m.field for m in report.mismatches)


def test_an_incompatible_grading_version_is_detected(small_run):
    _, manifest = small_run
    tampered = manifest.model_copy(update={"predicate_library_hash": "f" * 64})
    report = verify_manifest(tampered, FROZEN_DIR)
    assert not report.ok
    assert any("predicate_library_hash" in skew for skew in report.version_skew)


def test_a_missing_scenario_is_reported_not_skipped(small_run, tmp_path):
    """An unverifiable episode must not quietly count as verified."""
    _, manifest = small_run
    empty = tmp_path / "no_scenarios"
    empty.mkdir()
    report = verify_manifest(manifest, empty)
    assert not report.ok
    assert len(report.missing_scenarios) == len(manifest.episodes)
    assert report.episodes_checked == 0
