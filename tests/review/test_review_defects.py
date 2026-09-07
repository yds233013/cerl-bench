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


def test_an_empty_manifest_does_not_read_as_a_reproduced_result(tmp_path):
    """It passes every check, because there is nothing to check."""
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {"agent": {"name": "x", "kind": "model", "privilege_mode": "unprivileged"},
             "episodes": []},
        ),
    )
    report = verify_run.verify_manifest(manifest_module.load(path), freeze.FROZEN_DIR)
    assert report.ok
    assert "NO episodes" in report.summary()
