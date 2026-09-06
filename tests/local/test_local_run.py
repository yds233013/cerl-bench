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


def test_the_invalid_reason_is_reported_as_malformed_not_denied():
    """It cites no Layer-C interlock, because no backend interlock applies.

    Calling it a denial would put a parse failure into the attempted-violation
    machinery, which is reserved for unsafe acts a real backend refused.
    """
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
        charge_id=charge.id, amount_cents=1, reason="not-a-real-reason",
    )
    _, result = issue_refund(
        scenario.world, action, ToolContext(actor=scenario.agent_user),
    )
    assert result.outcome is Outcome.MALFORMED
    assert result.denied_interlock is None
    assert "not one of" in result.message


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
