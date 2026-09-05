"""The metered client, exercised end to end against a synthetic transport.

Every path a live run can take is covered here without a paid call: tool calls,
terminal turns, malformed turns, transport failures, retries, timeouts, and a
response that arrives with no usage to price it.
"""

from __future__ import annotations

import pytest

from cerl.agents.budget import MockTokenCounter, SpendLedger
from cerl.agents.model_client import (
    AmbiguousRequestOutcome,
    AnthropicClient,
    LiveEvaluationNotAuthorized,
    TranscriptCacheClient,
    TranscriptCacheMiss,
    live_evaluation_authorized,
)
from cerl.agents.synthetic_transport import (
    SyntheticFailure,
    SyntheticTimeout,
    SyntheticTransport,
    text_turn,
    tool_turn,
    usage_less_turn,
)

OPUS = "claude-opus-5"
TOOLS = [{"name": "tickets__get", "input_schema": {"type": "object"}}]


def build(steps, cap=10_000.0, max_retries=2):
    ledger = SpendLedger(cap_cents=cap, model=OPUS, counter_name="mock")
    transport = SyntheticTransport(steps)
    client = AnthropicClient(
        model=OPUS,
        max_tokens=2048,
        ledger=ledger,
        transport=transport,
        counter=MockTokenCounter(),
        max_retries=max_retries,
        require_authorization=False,
    )
    return client, ledger, transport


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_a_synthetic_transport_can_never_produce_a_live_label():
    """Provenance comes from the object that made the bytes, not from a flag."""
    client, _, _ = build([tool_turn("tickets__get", {"ticket_id": "tkt_000000000001"})])
    response = client.complete("sys", [], TOOLS)
    assert response.source == "synthetic"
    assert response.tool_name == "tickets__get"


def test_authorization_is_still_required_for_a_default_live_transport(monkeypatch):
    monkeypatch.delenv("CERL_LIVE_EVAL_AUTHORIZED", raising=False)
    monkeypatch.delenv("CERL_LIVE_EVAL_BUDGET_CENTS", raising=False)
    assert not live_evaluation_authorized()
    with pytest.raises(LiveEvaluationNotAuthorized):
        AnthropicClient()


def test_a_ledger_cannot_widen_the_authorised_cap(monkeypatch):
    monkeypatch.setenv("CERL_LIVE_EVAL_AUTHORIZED", "1")
    monkeypatch.setenv("CERL_LIVE_EVAL_BUDGET_CENTS", "100")
    with pytest.raises(LiveEvaluationNotAuthorized, match="exceeds the authorised"):
        AnthropicClient(ledger=SpendLedger(cap_cents=99_999.0, model=OPUS))


# --------------------------------------------------------------------------
# normal turns
# --------------------------------------------------------------------------


def test_a_tool_turn_is_confirmed_from_real_usage():
    client, ledger, _ = build(
        [tool_turn("tickets__get", {"ticket_id": "tkt_000000000001"},
                   input_tokens=1500, output_tokens=200)],
    )
    client.complete("sys", [], TOOLS)
    assert ledger.requests_confirmed == 1
    assert ledger.unresolved_cents == 0.0
    assert ledger.reserved_cents == pytest.approx(0.0)
    assert ledger.confirmed_cents > 0.0


def test_a_terminal_turn_is_handled_like_any_other():
    client, ledger, _ = build([tool_turn("finish", {"summary": "done"})])
    response = client.complete("sys", [], TOOLS)
    assert response.tool_name == "finish"
    assert ledger.requests_confirmed == 1


def test_a_turn_naming_no_tool_still_costs_and_is_returned():
    """A malformed turn is a real behaviour, and it is still billed."""
    client, ledger, _ = build([text_turn("I would refund the charge.")])
    response = client.complete("sys", [], TOOLS)
    assert response.tool_name is None
    assert ledger.confirmed_cents > 0.0


def test_the_whole_conversation_is_counted_each_request():
    counter = MockTokenCounter()
    ledger = SpendLedger(cap_cents=10_000.0, model=OPUS)
    client = AnthropicClient(
        model=OPUS, ledger=ledger,
        transport=SyntheticTransport(default=tool_turn("finish", {"summary": "x"})),
        counter=counter, require_authorization=False,
    )
    client.complete("sys", [{"role": "user", "content": "a"}], TOOLS)
    first = ledger.confirmed_cents
    client.complete("sys", [{"role": "user", "content": "a"}] * 20, TOOLS)
    assert counter.calls == 2
    # usage is scripted identically, so growth shows up in the reservation
    assert ledger.confirmed_cents > first


# --------------------------------------------------------------------------
# failures, retries, and ambiguity
# --------------------------------------------------------------------------


def test_a_failure_is_retried_and_every_attempt_is_charged():
    """A retry is another billable request, so it reserves in its own right."""
    client, ledger, transport = build(
        [SyntheticFailure("500"), SyntheticFailure("500"),
         tool_turn("finish", {"summary": "ok"})],
    )
    client.complete("sys", [], TOOLS)
    assert transport.index == 3
    assert ledger.requests_confirmed == 1
    # the two failed attempts are unresolved, not forgiven
    assert ledger.requests_unresolved == 2
    assert ledger.unresolved_cents > 0.0


def test_exhausting_retries_raises_and_leaves_the_charges_standing():
    client, ledger, _ = build([SyntheticFailure("500")] * 3, max_retries=2)
    with pytest.raises(AmbiguousRequestOutcome):
        client.complete("sys", [], TOOLS)
    assert ledger.requests_unresolved == 3
    assert ledger.confirmed_cents == 0.0


def test_a_timeout_is_charged_because_it_may_have_reached_the_provider():
    client, ledger, _ = build(
        [SyntheticTimeout("read timeout"), tool_turn("finish", {"summary": "ok"})],
    )
    client.complete("sys", [], TOOLS)
    assert ledger.requests_unresolved == 1
    assert any("Timeout" in r for r in ledger.unresolved_reasons)


def test_a_response_without_usage_is_charged_not_released():
    """It certainly cost money; we simply cannot say how much."""
    client, ledger, _ = build([usage_less_turn("finish", {"summary": "ok"})])
    response = client.complete("sys", [], TOOLS)
    assert response.tool_name == "finish"
    assert ledger.requests_confirmed == 0
    assert ledger.requests_unresolved == 1
    assert ledger.unresolved_reasons == ["response carried no usage"]


def test_the_budget_stops_the_run_rather_than_overrunning():
    from cerl.agents.budget import BudgetExceeded

    client, ledger, _ = build([tool_turn("finish", {"summary": "ok"})] * 50, cap=12.0)
    sent = 0
    stopped = False
    for _ in range(50):
        try:
            client.complete("sys", [], TOOLS)
        except BudgetExceeded:
            stopped = True
            break
        sent += 1
    assert stopped, "the run was never stopped by the budget"
    assert sent >= 1
    assert ledger.committed_cents <= ledger.cap_cents


def test_sdk_retries_are_disabled_so_no_request_is_unreserved():
    """An SDK-level automatic retry would be a request the ledger never saw.

    The transport constructor takes max_retries=0 for exactly this reason; the
    default is asserted here so a future change to it fails loudly.
    """
    import inspect

    from cerl.agents import model_client

    signature = inspect.signature(model_client.AnthropicTransport.__init__)
    assert signature.parameters["max_retries"].default == 0
    assert SyntheticTransport.source == "synthetic"


# --------------------------------------------------------------------------
# the cache client
# --------------------------------------------------------------------------


def test_a_cache_miss_fails_rather_than_reaching_the_network():
    client = TranscriptCacheClient({})
    with pytest.raises(TranscriptCacheMiss):
        client.complete("sys", [], TOOLS)
