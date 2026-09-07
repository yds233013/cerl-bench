"""Regressions for the five defects found by independent review of fc2a301.

Each test reproduces the reported failure, so it fails on the old code for the
reported reason rather than passing incidentally.
"""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from cerl.actions import Action, TicketsGet
from cerl.core import TicketId
from cerl.env.env import CerlEnv
from cerl.eval import manifest as manifest_module
from cerl.eval import verify_run
from cerl.scenario import freeze


@pytest.fixture(scope="module")
def w2(all_frozen):
    return [s for s in all_frozen if s.family == "duplicate_charge_approval"]


# --------------------------------------------------------------------------
# 1. manifest verification must compare every replay-derivable field
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recorded():
    from tests.helpers import REPO

    return json.loads(
        (REPO / "evidence/local-baseline/local_run.json").read_text(encoding="utf-8"),
    )


def _verify(payload: dict, tmp_path) -> verify_run.VerificationReport:
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))
    return verify_run.verify_manifest(manifest_module.load(path), freeze.FROZEN_DIR)


def test_the_untouched_historical_run_still_verifies(recorded, tmp_path):
    """The fix must not make old evidence unverifiable."""
    report = _verify(copy.deepcopy(recorded), tmp_path)
    assert report.ok, report.summary()
    assert report.episodes_checked == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_completion", 1.0),
        ("correct_final_state", True),
        ("decision_correct", True),
        ("truncated", True),
        ("tool_calls", 999),
        ("declared_outcome", "finish"),
    ],
)
def test_an_edited_verdict_field_fails_verification(recorded, tmp_path, field, value):
    """Each of these was previously unchecked; a manifest could claim anything."""
    altered = copy.deepcopy(recorded)
    for episode in altered["episodes"]:
        episode[field] = value
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    assert any(field in m.field for m in report.mismatches), report.summary()


def test_inflated_results_with_consistent_aggregates_still_fail(recorded, tmp_path):
    """The reported reproduction: aggregates recomputed from altered records.

    Aggregates are now derived from the *replayed* records, so making the file
    internally consistent no longer helps.
    """
    from cerl.eval.metrics import aggregate

    altered = copy.deepcopy(recorded)
    for episode in altered["episodes"]:
        episode["task_completion"] = 1.0
        episode["correct_final_state"] = True
        episode["decision_correct"] = True
    altered.pop("manifest_hash", None)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(altered))
    manifest = manifest_module.load(path)
    manifest = manifest.model_copy(
        update={"metrics": type(manifest.metrics)(aggregate(list(manifest.episodes)))},
    )
    report = verify_run.verify_manifest(manifest, freeze.FROZEN_DIR)
    assert not report.ok, "consistent aggregates must not launder altered episodes"


def test_a_corrupted_step_fails_even_with_its_original_entry_hash(recorded, tmp_path):
    """The reported probe: outcome, state hash and violation classes rewritten."""
    altered = copy.deepcopy(recorded)
    step = altered["episodes"][0]["steps"][0]
    step["outcome"] = "invented-outcome"
    step["state_hash_after"] = "f" * 64
    step["committed_classes"] = ["C_DISCLOSE"]
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    fields = " ".join(m.field for m in report.mismatches)
    assert "step[0]" in fields, report.summary()


def test_the_manifest_checksum_is_validated_separately(recorded, tmp_path):
    """It catches corruption. It is not evidence of honesty, and does not claim to be."""
    path = tmp_path / "m.json"
    payload = copy.deepcopy(recorded)
    payload["manifest_hash"] = "0" * 64
    path.write_text(json.dumps(payload))
    result = manifest_module.checksum_of(path)
    assert result.present and not result.valid

    report = verify_run.verify_manifest(
        manifest_module.load(path), freeze.FROZEN_DIR, path,
    )
    assert not report.ok
    assert "MISMATCH" in report.checksum


def test_a_manifest_without_a_checksum_is_accepted_explicitly(tmp_path):
    """Historical files predate the field; that is reported, not failed."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"agent": {"name": "x", "kind": "model",
                                          "privilege_mode": "unprivileged"}}))
    result = manifest_module.checksum_of(path)
    assert not result.present
    assert result.valid
    assert "absent" in result.status


def test_verification_states_what_replay_cannot_prove():
    assert verify_run.UNVERIFIABLE_CLAIMS
    joined = " ".join(verify_run.UNVERIFIABLE_CLAIMS)
    assert "model" in joined and "produced" in joined


# --------------------------------------------------------------------------
# 2. nested payload aliasing must not reach the sealed trace
# --------------------------------------------------------------------------


def test_mutating_an_observation_payload_cannot_break_the_sealed_trace(w2):
    """The reported reproduction, through the public observation only."""
    scenario = w2[0]
    env = CerlEnv(scenario)
    env.reset()
    result = env.step(TicketsGet(ticket_id=TicketId(str(scenario.variables["ticket"]))))
    env.world.trace.verify_chain()

    result.observation.result.payload["ticket"]["status"] = "mutated-by-consumer"

    assert (
        env.world.trace.entries[0].result.payload["ticket"]["status"]
        != "mutated-by-consumer"
    )
    env.world.trace.verify_chain()  # raises ChainBroken if the seal was damaged


def test_mutating_a_nested_list_in_a_payload_cannot_reach_the_trace(w2):
    """Lists as well as dictionaries, as the review asked."""
    scenario = w2[0]
    env = CerlEnv(scenario)
    env.reset()
    result = env.step(TicketsGet(ticket_id=TicketId(str(scenario.variables["ticket"]))))
    payload = result.observation.result.payload
    ticket = payload.get("ticket")
    if isinstance(ticket, dict) and isinstance(ticket.get("comments"), list):
        ticket["comments"].append({"author": "intruder", "text": "x"})
    sealed = env.world.trace.entries[0].result.payload.get("ticket")
    if isinstance(sealed, dict) and isinstance(sealed.get("comments"), list):
        assert all(c.get("author") != "intruder" for c in sealed["comments"])
    env.world.trace.verify_chain()


def test_as_document_does_not_hand_out_its_cache(w2):
    """Editing a returned document must not change later reads or the hash."""
    world = w2[0].world
    before_hash = world.state_hash()
    document = world.as_document()
    customer_id = next(iter(document["billing"]["customers"]))
    original = document["billing"]["customers"][customer_id]["display_name"]
    document["billing"]["customers"][customer_id]["display_name"] = "MUTATED"

    again = world.as_document()
    assert again["billing"]["customers"][customer_id]["display_name"] == original
    assert world.state_hash() == before_hash


def test_deep_freeze_rejects_nested_mutation():
    from cerl.core import deep_freeze

    frozen = deep_freeze({"a": {"b": [1, {"c": 2}]}})
    with pytest.raises((TypeError, AttributeError)):
        frozen["a"]["b"][1]["c"] = 9  # type: ignore[index]
    with pytest.raises(AttributeError):
        frozen["a"]["b"].append(3)  # type: ignore[union-attr]


def test_replay_hashes_are_unchanged_by_the_isolation(w2):
    """Canonical serialisation must be untouched, or history stops replaying."""
    from cerl.reference.registry import oracle_for
    from cerl.reference.runner import run_reference

    episode = run_reference(w2[0], oracle_for(w2[0]))
    episode.trace.verify_chain()
    assert episode.verdict.safe_completion


# --------------------------------------------------------------------------
# 4. finite-domain action fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [0, -1, -42000])
def test_non_positive_refunds_are_rejected_before_dispatch(w2, amount):
    """They validated, then raised inside the Refund constructor, ending the run."""
    charge = str(w2[0].variables["target_charge"])
    with pytest.raises(ValidationError):
        TypeAdapter(Action).validate_python(
            {
                "kind": "billing.issue_refund",
                "charge_id": charge,
                "amount_cents": amount,
                "reason": "duplicate",
            },
        )


def test_a_positive_refund_still_works(w2):
    """The bound must not have broken the ordinary path."""
    scenario = w2[0]
    env = CerlEnv(scenario)
    env.reset()
    action = TypeAdapter(Action).validate_python(
        {
            "kind": "billing.issue_refund",
            "charge_id": str(scenario.variables["target_charge"]),
            "amount_cents": int(scenario.variables["duplicate_amount"]),
            "reason": "duplicate",
        },
    )
    env.step(action)
    assert len(env.world.trace.entries) == 1


def test_the_public_schema_advertises_the_bound_and_the_closed_sets():
    """A handler-only fix would leave the advertised contract still wrong."""
    from cerl.agents.tool_schemas import all_tool_schemas

    schemas = {t["name"]: t["input_schema"]["properties"] for t in all_tool_schemas()}
    assert schemas["billing__issue_refund"]["amount_cents"]["exclusiveMinimum"] == 0
    assert schemas["tickets__set_status"]["status"]["enum"] == [
        "open", "pending_customer", "resolved", "escalated",
    ]
    assert schemas["tickets__add_comment"]["comment_kind"]["enum"] == [
        "resolution", "escalation", "info_request", "note",
    ]


def test_an_unknown_comment_kind_is_rejected_not_silently_renamed(w2):
    """It used to be recorded as `note`, which is not what the caller asked for."""
    with pytest.raises(ValidationError):
        TypeAdapter(Action).validate_python(
            {
                "kind": "tickets.add_comment",
                "ticket_id": str(w2[0].variables["ticket"]),
                "text": "x",
                "comment_kind": "not-a-real-kind",
            },
        )


def test_the_dispatcher_scores_a_bad_amount_as_malformed(w2):
    """The agent path, not just the type adapter."""
    from cerl.agents.model_client import ModelResponse
    from cerl.agents.prompt_only import PromptOnlyAgent
    from cerl.core import FrozenMap

    agent = PromptOnlyAgent.__new__(PromptOnlyAgent)
    action = PromptOnlyAgent._parse(
        agent,
        ModelResponse(
            tool_name="billing__issue_refund",
            tool_input=FrozenMap(
                {
                    "charge_id": str(w2[0].variables["target_charge"]),
                    "amount_cents": 0,
                    "reason": "duplicate",
                },
            ),
            source="local",
        ),
    )
    assert action.kind == "malformed"


def test_the_tool_schema_version_records_the_change():
    from cerl.agents.tool_schemas import TOOL_SCHEMA_VERSION

    assert TOOL_SCHEMA_VERSION == "1.2.0"


_EMPTY_AGENT = {"name": "x", "kind": "model", "privilege_mode": "unprivileged"}


def test_an_empty_manifest_does_not_read_as_a_reproduced_result(tmp_path):
    """It passes every check there is to run, which is not the same as evidence.

    The metric block still has to be present and correct -- for zero episodes
    that is the zero-valued block ``aggregate([])`` produces -- because "no
    episodes" is not a licence to omit numbers.
    """
    from cerl.eval.metrics import aggregate

    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {"agent": _EMPTY_AGENT, "episodes": [], "metrics": aggregate([])},
        ),
    )
    report = verify_run.verify_manifest(manifest_module.load(path), freeze.FROZEN_DIR)
    assert report.ok, report.summary()
    assert "NO episodes" in report.summary()


def test_an_empty_manifest_still_has_to_carry_its_metric_block(tmp_path):
    """Omitting it entirely is a missing metric, not an empty run."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"agent": _EMPTY_AGENT, "episodes": []}))
    report = verify_run.verify_manifest(manifest_module.load(path), freeze.FROZEN_DIR)
    assert not report.ok
    assert "metrics.episodes" in {m.field for m in report.mismatches}


# --------------------------------------------------------------------------
# 1b. the metric block is compared exactly, key set included
# --------------------------------------------------------------------------


def _metric_fields(report: verify_run.VerificationReport) -> set[str]:
    return {m.field for m in report.mismatches if m.field.startswith("metrics.")}


def test_an_empty_metric_block_fails_rather_than_being_skipped(recorded, tmp_path):
    """Deleting the whole block used to be free: absent read as nothing to check."""
    altered = copy.deepcopy(recorded)
    altered["metrics"] = {}
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    fields = _metric_fields(report)
    assert "metrics.safe_completion_rate" in fields
    assert "metrics.committed_violation_rate" in fields
    assert "metrics.attempted_violation_rate" in fields
    assert all(
        m.recorded == "<absent from the manifest>"
        for m in report.mismatches
        if m.field in fields
    )


@pytest.mark.parametrize(
    "metric",
    [
        "safe_completion_rate",
        "task_completion_mean",
        "committed_violation_rate",
        "attempted_violation_rate",
        "committed_by_class",
        "attempted_by_class",
        "episodes",
    ],
)
def test_a_single_missing_metric_is_caught_and_named(recorded, tmp_path, metric):
    altered = copy.deepcopy(recorded)
    del altered["metrics"][metric]
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    assert _metric_fields(report) == {f"metrics.{metric}"}, report.summary()


def test_an_invented_metric_is_caught_and_named(recorded, tmp_path):
    """Nothing looked at keys the run does not produce, so one could be asserted."""
    altered = copy.deepcopy(recorded)
    altered["metrics"]["safety_score"] = 0.99
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    assert _metric_fields(report) == {"metrics.safety_score"}
    invented = next(m for m in report.mismatches if m.field == "metrics.safety_score")
    assert invented.recomputed == "<not a metric this run produces>"


def test_a_changed_metric_value_names_that_metric_alone(recorded, tmp_path):
    altered = copy.deepcopy(recorded)
    altered["metrics"]["safe_completion_rate"] = 1.0
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    assert _metric_fields(report) == {"metrics.safe_completion_rate"}


def test_the_two_violation_series_are_compared_separately(recorded, tmp_path):
    """Swapping them must fail, not cancel out.

    A verifier that compared their sum -- or compared only one -- would accept
    this. The separation is CLAUDE.md rule 2, and it has to hold in the checker
    as much as in the recorder.
    """
    altered = copy.deepcopy(recorded)
    metrics = altered["metrics"]
    metrics["committed_violation_rate"], metrics["attempted_violation_rate"] = (
        metrics["attempted_violation_rate"],
        metrics["committed_violation_rate"],
    )
    metrics["committed_by_class"], metrics["attempted_by_class"] = (
        metrics["attempted_by_class"],
        metrics["committed_by_class"],
    )
    altered.pop("manifest_hash", None)
    report = _verify(altered, tmp_path)
    assert not report.ok
    assert "metrics.committed_violation_rate" in _metric_fields(report)
    assert "metrics.committed_by_class" in _metric_fields(report)


def test_a_non_replayable_metric_is_exempt_by_an_explicit_allowlist(recorded, tmp_path):
    """Provider spend has nothing to recompute against; everything else does."""
    assert "spend" in verify_run.UNVERIFIABLE_METRICS
    altered = copy.deepcopy(recorded)
    altered["metrics"]["spend"] = {"cap_cents": 100000.0, "confirmed_cents": 90.0}
    altered.pop("manifest_hash", None)
    assert _verify(altered, tmp_path).ok, "a live pilot's spend block must not fail"


def test_the_allowlist_matches_what_the_code_actually_writes():
    """Guessed exemptions are worse than none: they read as cover and are not.

    Two places write manifest metrics -- ``aggregate()`` and the pilot, which
    attaches the ledger report under ``spend``. Every key one of them emits must
    be either recomputed or explicitly exempt, and nothing may be exempt that
    nothing emits.
    """
    from cerl.eval.metrics import aggregate

    replayable = set(aggregate([]))
    written_by_the_pilot = {"spend"}  # cerl/eval/pilot.py, _stamp
    emitted = replayable | written_by_the_pilot

    assert emitted >= verify_run.UNVERIFIABLE_METRICS, (
        "the allowlist names a metric nothing writes: "
        f"{sorted(verify_run.UNVERIFIABLE_METRICS - emitted)}"
    )
    assert emitted - replayable == verify_run.UNVERIFIABLE_METRICS, (
        "a metric is emitted that is neither recomputed nor exempt: "
        f"{sorted(emitted - replayable - verify_run.UNVERIFIABLE_METRICS)}"
    )


def test_the_allowlist_covers_only_what_replay_cannot_produce(recorded):
    """It must never shadow a metric the run actually computes."""
    from cerl.eval.metrics import aggregate

    assert verify_run.UNVERIFIABLE_METRICS.isdisjoint(set(aggregate([])))
