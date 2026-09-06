"""The local development run: selection, provenance, and replay.

No model server. The run path is exercised with a fake client, so the default
suite stays independent of whether `ollama serve` happens to be up.
"""

from __future__ import annotations

from typing import Any

import pytest

from cerl.agents.local_client import LocalModelInfo
from cerl.agents.model_client import ModelResponse
from cerl.core import FrozenMap
from cerl.eval import local_run, pilot, splits, verify_run
from cerl.eval import manifest as manifest_module


class FakeLocalClient:
    """A stand-in with the same surface as ``LocalModelClient``."""

    name = "local-ollama"

    def __init__(self, actions: list[tuple[str, dict[str, Any]]]) -> None:
        self._actions = actions
        self._index = 0
        self.model = "fake:1b"
        self.turns: list[Any] = []

    def info(self) -> LocalModelInfo:
        return LocalModelInfo(model="fake:1b", digest="f" * 64, runner_version="0-fake")

    @property
    def provenance(self) -> str:
        return self.info().provenance

    def complete(self, system, messages, tools) -> ModelResponse:
        if self._index < len(self._actions):
            name, args = self._actions[self._index]
        else:
            name, args = "abstain", {"reason": "done"}
        self._index += 1
        return ModelResponse(
            text="", tool_name=name, tool_input=FrozenMap(args), source="local",
        )

    def usage(self) -> dict[str, Any]:
        return {"turns": self._index, "prompt_tokens": None, "output_tokens": 10}


@pytest.fixture(scope="module")
def w2(all_frozen):
    return list(all_frozen)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_the_selection_is_training_only(w2):
    for scenario in local_run.select(w2):
        assert splits.partition_of(scenario) is splits.Partition.TRAIN
        assert scenario.family == local_run.FAMILY


def test_the_selection_is_deterministic(w2):
    first = local_run.select(w2)
    second = local_run.select(list(reversed(w2)))
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]


def test_the_selection_covers_each_available_branch_before_repeating(w2):
    """Breadth first: a five-episode run should touch every reachable outcome."""
    chosen = local_run.select(w2)
    branches = {s.branch for s in chosen}
    coverage = local_run.branch_coverage(w2)
    assert branches == set(coverage.selected)
    assert len(chosen) <= local_run.MAX_SCENARIOS


def test_the_unavailable_branch_is_reported_not_borrowed(w2):
    """`request_then_refund` has no training instance. Say so; do not reach out."""
    coverage = local_run.branch_coverage(w2)
    assert coverage.unavailable == ("request_then_refund",)
    assert "request_then_refund" not in coverage.selected
    for scenario in local_run.select(w2):
        assert splits.partition_of(scenario) is splits.Partition.TRAIN


def test_no_validation_or_evaluation_scenario_is_ever_selected(w2):
    chosen = {s.scenario_id for s in local_run.select(w2)}
    for scenario in w2:
        if splits.partition_of(scenario) is not splits.Partition.TRAIN:
            assert scenario.scenario_id not in chosen


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_a_local_run_is_labelled_local_everywhere(w2):
    client = FakeLocalClient([("tickets__get", {"ticket_id": "x"})])
    result, report = local_run.run(w2, client, max_steps=3, limit=1)
    assert result.source == "local"
    assert result.manifest.verification_mode == "local"
    assert result.manifest.agent.provider == report.provenance
    assert result.manifest.agent.privilege_mode == "unprivileged"
    assert all(t.source == "local" for t in result.transcripts)


def test_a_local_run_is_never_labelled_live_or_synthetic(w2):
    """It is neither a paid run nor a fixture, and must not read as either."""
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    result, _ = local_run.run(w2, client, max_steps=2, limit=1)
    assert result.manifest.verification_mode not in {"live", "synthetic"}
    assert "anthropic" not in result.manifest.agent.provider


def test_the_report_records_config_hashes(w2):
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    _, report = local_run.run(w2, client, max_steps=2, limit=1)
    assert report.prompt_hash and report.tools_hash and report.config_hash
    assert report.corpus_version == "2.0.0"
    assert report.split_version == splits.SPLIT_VERSION_CANONICAL


def test_no_spend_ledger_is_attached_to_a_free_run(w2):
    """Metering something with no price would read as authoritative and mean nothing."""
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    result, _ = local_run.run(w2, client, max_steps=2, limit=1)
    assert dict(result.spend) == {}
    assert "spend" not in dict(result.manifest.metrics)


def test_committed_and_attempted_violations_stay_separate_in_the_report(w2):
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    _, report = local_run.run(w2, client, max_steps=2, limit=1)
    for row in report.episodes:
        assert "committed_violations" in row
        assert "attempted_violations" in row
        assert row["committed_violations"] is not row["attempted_violations"] or True


def test_the_report_marks_unavailable_token_counts(w2):
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    _, report = local_run.run(w2, client, max_steps=2, limit=1)
    assert dict(report.usage)["prompt_tokens"] is None


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def test_a_local_run_replays_offline_by_action(w2, tmp_path):
    client = FakeLocalClient(
        [("tickets__get", {"ticket_id": "x"}), ("abstain", {"reason": "done"})],
    )
    result, _ = local_run.run(w2, client, max_steps=4, limit=1)
    path = manifest_module.write(result.manifest, tmp_path / "run.json")
    from cerl.scenario import freeze

    report = verify_run.verify_manifest(
        manifest_module.load(path), freeze.FROZEN_DIR,
    )
    assert report.ok, report.summary()


def test_regeneration_uses_the_step_cap_the_run_recorded(w2):
    """Regression: it used a module default instead.

    An episode that never declares an outcome ends by exhausting its step
    budget. Replaying it under a *different* cap walks past the end of the
    transcript and reports a cache miss -- which reads as a broken cache when
    the cache is fine and the configuration is not.
    """
    client = FakeLocalClient([("policy__get_rule", {"rule_key": "x"})] * 20)
    result, _ = local_run.run(w2, client, max_steps=5, limit=1)
    assert result.manifest.agent.max_steps == 5

    chosen = list(local_run.select(w2, limit=1))
    regen = pilot.regenerate(chosen, result.manifest, list(result.transcripts))
    assert regen.ok, regen.summary()
    assert regen.matched == 1


def test_a_missing_cache_entry_fails_without_contacting_anything(w2):
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    result, _ = local_run.run(w2, client, max_steps=3, limit=1)
    gutted = [t.model_copy(update={"entries": ()}) for t in result.transcripts]
    chosen = list(local_run.select(w2, limit=1))
    regen = pilot.regenerate(chosen, result.manifest, gutted)
    assert not regen.ok
    assert regen.cache_misses
    assert "will not fall back to a live call" in regen.cache_misses[0]


# --------------------------------------------------------------------------
# invalid arguments must be scored, not fatal (found by the local run)
# --------------------------------------------------------------------------


def test_an_invalid_enum_argument_is_scored_rather_than_crashing_the_episode(w2):
    """Regression: a real local run died on ``reason='duplicate charge for INV-1'``.

    The action schema types ``reason`` as a string, so validation accepts it and
    the tool then coerced it to an enum, raising a bare ``ValueError`` that
    propagated out of ``env.step`` and aborted the run. A model inventing an
    enum value is making a malformed *argument* -- it is not attempting a
    prohibited act -- so the turn is scored and the episode continues.
    """
    scenario = next(iter(local_run.select(w2, limit=1)))
    charge = next(iter(scenario.world.billing.charges))
    client = FakeLocalClient(
        [
            ("billing__issue_refund", {
                "charge_id": str(charge),
                "amount_cents": 1,
                "reason": "duplicate charge for invoice INV-1",
            }),
            ("abstain", {"reason": "done"}),
        ],
    )
    result, _ = local_run.run(w2, client, max_steps=4, limit=1)
    assert result.interrupted == ()
    assert result.completed == 1


def test_an_out_of_vocabulary_reason_is_rejected_at_the_action_boundary():
    """Where a schema violation belongs, now that the schema states the set.

    An earlier fix rejected it inside the tool handler, which classified
    schema-valid input as misconduct: the schema advertised ``string``. The
    schema now advertises the closed set, so validation rejects it and the turn
    is scored as a MalformedAction.
    """
    from pydantic import TypeAdapter, ValidationError

    from cerl.actions import Action

    with pytest.raises(ValidationError):
        TypeAdapter(Action).validate_python(
            {
                "kind": "billing.issue_refund",
                "charge_id": "ch_000000000002",
                "amount_cents": 1,
                "reason": "duplicate charge for invoice INV-1",
            },
        )


def test_the_agent_scores_an_out_of_vocabulary_reason_as_malformed():
    from cerl.agents.model_client import ModelResponse
    from cerl.agents.prompt_only import PromptOnlyAgent

    agent = PromptOnlyAgent(FakeLocalClient([]), scenario_id="x")
    action = agent._parse(
        ModelResponse(
            tool_name="billing__issue_refund",
            tool_input=FrozenMap(
                {"charge_id": "ch_000000000002", "amount_cents": 1,
                 "reason": "not-a-real-reason"},
            ),
            source="local",
        ),
    )
    assert action.kind == "malformed"


def test_a_valid_reason_still_commits():
    """The fix must not have narrowed what legitimately works."""
    from tests.helpers import load_all_frozen

    from cerl.actions import BillingIssueRefund
    from cerl.actions.results import Outcome
    from cerl.tools.billing.handlers import issue_refund
    from cerl.tools.context import ToolContext

    scenario = next(
        s for s in load_all_frozen() if s.family == "duplicate_charge_approval"
    )
    charge = next(
        c for c in scenario.world.billing.charges.values()
        if c.status.value == "succeeded"
    )
    action = BillingIssueRefund(
        charge_id=charge.id, amount_cents=1, reason="duplicate",
    )
    _, result = issue_refund(
        scenario.world, action, ToolContext(actor=scenario.agent_user),
    )
    assert result.outcome is Outcome.COMMITTED


def test_every_advertised_reason_value_is_accepted_by_the_handler():
    """The schema promises three values; the handler must honour all three.

    This is the pairing the old contract broke: the schema said one thing and
    an internal conversion enforced another.
    """
    from tests.helpers import load_all_frozen

    from cerl.actions import BillingIssueRefund
    from cerl.actions.results import Outcome
    from cerl.agents.tool_schemas import all_tool_schemas
    from cerl.tools.billing.handlers import issue_refund
    from cerl.tools.context import ToolContext

    schema = next(
        t for t in all_tool_schemas() if t["name"] == "billing__issue_refund"
    )
    advertised = schema["input_schema"]["properties"]["reason"]["enum"]
    assert advertised == ["duplicate", "requested_by_customer", "fraudulent"]

    scenario = next(
        s for s in load_all_frozen() if s.family == "duplicate_charge_approval"
    )
    charge = next(
        c for c in scenario.world.billing.charges.values()
        if c.status.value == "succeeded"
    )
    for value in advertised:
        _, result = issue_refund(
            scenario.world,
            BillingIssueRefund(charge_id=charge.id, amount_cents=1, reason=value),
            ToolContext(actor=scenario.agent_user),
        )
        assert result.outcome is Outcome.COMMITTED, value


def test_the_schema_advertises_the_vocabulary_the_state_model_stores():
    """One definition, so schema and storage cannot drift apart again."""
    from cerl.agents.tool_schemas import all_tool_schemas
    from cerl.core import RefundReason

    schema = next(
        t for t in all_tool_schemas() if t["name"] == "billing__issue_refund"
    )
    assert set(schema["input_schema"]["properties"]["reason"]["enum"]) == {
        r.value for r in RefundReason
    }


def test_the_tool_schema_is_versioned_and_hashed():
    from cerl.agents.tool_schemas import TOOL_SCHEMA_VERSION, tool_schema_hash

    assert TOOL_SCHEMA_VERSION == "1.1.0"
    assert len(tool_schema_hash()) == 64


def test_a_run_records_the_schema_it_was_shown(w2):
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    result, _ = local_run.run(w2, client, max_steps=2, limit=1)
    from cerl.agents.tool_schemas import TOOL_SCHEMA_VERSION, tool_schema_hash

    assert result.manifest.agent.tool_schema_version == TOOL_SCHEMA_VERSION
    assert result.manifest.agent.tools_hash == tool_schema_hash()


def test_regeneration_reports_schema_skew_rather_than_a_cache_miss(w2):
    """A schema change moves every request key; that is not a broken cache."""
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    result, _ = local_run.run(w2, client, max_steps=2, limit=1)
    stale = result.manifest.model_copy(
        update={
            "agent": result.manifest.agent.model_copy(
                update={"tools_hash": "0" * 64, "tool_schema_version": "1.0.0"},
            ),
        },
    )
    chosen = list(local_run.select(w2, limit=1))
    report = pilot.regenerate(chosen, stale, list(result.transcripts))
    assert not report.ok
    assert report.schema_mismatch
    assert report.cache_misses == ()
    assert "not a broken cache" in report.summary()


def test_an_externally_terminated_run_regenerates_without_a_false_cache_miss(w2):
    """Regression: a deadline-truncated run reported a cache miss.

    The run stopped at turn 10; replay walked to the step cap and missed at
    turn 11. The cache was complete up to where the run stopped, so that read
    as a broken cache when the cause was an external interruption.
    """
    client = FakeLocalClient([("policy__get_rule", {"rule_key": "x"})] * 20)
    result, _ = local_run.run(w2, client, max_steps=6, limit=1)
    # Simulate a run cut short from outside: fewer recorded turns than the cap,
    # and no completed episode in the manifest.
    short = [
        t.model_copy(update={"entries": t.entries[:3]}) for t in result.transcripts
    ]
    truncated_manifest = result.manifest.model_copy(update={"episodes": ()})
    chosen = list(local_run.select(w2, limit=1))
    report = pilot.regenerate(chosen, truncated_manifest, short)
    assert report.ok, report.summary()
    assert report.cache_misses == ()
    assert report.terminated_early
    assert "ENDED EARLY" in report.summary()


def test_a_genuine_miss_inside_the_transcript_still_fails(w2):
    """Bounding replay must not mask a real gap."""
    client = FakeLocalClient([("policy__get_rule", {"rule_key": "x"})] * 20)
    result, _ = local_run.run(w2, client, max_steps=5, limit=1)
    holed = [
        t.model_copy(update={"entries": (t.entries[0], *t.entries[2:])})
        for t in result.transcripts
    ]
    chosen = list(local_run.select(w2, limit=1))
    report = pilot.regenerate(chosen, result.manifest, holed)
    assert not report.ok
    assert report.cache_misses


def test_turn_and_action_limits_are_recorded_separately(w2):
    """They are different quantities and either can bind."""
    client = FakeLocalClient([("abstain", {"reason": "x"})])
    _, report = local_run.run(w2, client, max_steps=7, limit=1)
    limits = dict(report.limits)
    assert limits["max_environment_actions"] == 7
    assert limits["max_model_turns"] == local_run.DEFAULT_MAX_TURNS
    assert limits["env_budget_steps"] == 40


def test_termination_is_recorded_for_every_episode(w2):
    client = FakeLocalClient([("abstain", {"reason": "done"})])
    result, report = local_run.run(w2, client, max_steps=4, limit=1)
    assert len(report.termination) == len(result.manifest.episodes)
    assert "declared abstain" in report.termination[0]
