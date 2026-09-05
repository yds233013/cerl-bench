"""Spend accounting: four quantities, kept apart, bounded during execution.

The module under test makes no dollar *guarantee* -- see
``docs/budget-accounting.md`` -- so these tests assert the operational limit it
does provide: no request is sent without fitting in the spendable balance, an
ambiguous outcome is charged rather than forgiven, and a resumed run cannot
grant itself the allowance again.
"""

from __future__ import annotations

import json
import threading

import pytest

from cerl.agents.budget import (
    API_MARGIN,
    HEURISTIC_MARGIN,
    PRICING,
    ApiTokenCounter,
    BudgetExceeded,
    HeuristicTokenCounter,
    MockTokenCounter,
    SpendLedger,
    UnpricedModel,
    cost_cents,
    resolve_counter,
)

OPUS = "claude-opus-5"


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------


def test_pricing_arithmetic():
    assert cost_cents(OPUS, 1_000_000, 0) == pytest.approx(500.0)
    assert cost_cents(OPUS, 0, 1_000_000) == pytest.approx(2500.0)


def test_cache_modifiers_are_priced_not_ignored():
    """Cache reads and writes are billed at different rates; both must count."""
    read = cost_cents(OPUS, 0, 0, cache_read_tokens=1_000_000)
    write = cost_cents(OPUS, 0, 0, cache_write_tokens=1_000_000)
    plain = cost_cents(OPUS, 1_000_000, 0)
    assert read == pytest.approx(50.0)
    assert write == pytest.approx(625.0)
    assert read < plain < write


def test_an_unpriced_model_raises_rather_than_costing_zero():
    """A model that appears free would run entirely uncapped."""
    with pytest.raises(UnpricedModel):
        cost_cents("some-unreleased-model", 1, 1)


def test_every_priced_model_has_four_rates():
    for model, rates in PRICING.items():
        assert len(rates) == 4, model
        assert all(r > 0 for r in rates), model


# --------------------------------------------------------------------------
# token counting
# --------------------------------------------------------------------------


def test_counters_count_the_whole_request_not_just_the_last_message():
    """Counting only the newest turn under-reserves by the whole transcript."""
    for counter in (HeuristicTokenCounter(), MockTokenCounter()):
        tools = [{"name": "t", "input_schema": {}}]
        short = counter.count("sys", [{"role": "user", "content": "hi"}], tools)
        long = counter.count(
            "sys",
            [{"role": "user", "content": "hi"}] * 12,
            tools,
        )
        assert long > short, counter.name


def test_the_api_counter_is_preferred_when_the_provider_offers_one():
    class _Messages:
        def count_tokens(self, **kwargs):
            class _R:
                input_tokens = 4242

            return _R()

    class _Client:
        messages = _Messages()

    counter = resolve_counter(_Client(), OPUS)
    assert isinstance(counter, ApiTokenCounter)
    assert counter.count("s", [], []) == 4242
    assert counter.uncertainty_margin == API_MARGIN


def test_the_heuristic_is_the_documented_fallback_with_a_wider_margin():
    """The fallback is allowed, but it must declare more uncertainty."""
    counter = resolve_counter(None, OPUS)
    assert isinstance(counter, HeuristicTokenCounter)
    assert counter.uncertainty_margin == HEURISTIC_MARGIN > API_MARGIN


# --------------------------------------------------------------------------
# the four quantities
# --------------------------------------------------------------------------


def test_estimated_reserved_confirmed_and_unresolved_stay_distinct():
    led = SpendLedger(cap_cents=1000.0, model=OPUS)
    assert led.committed_cents == 0.0

    held = led.reserve(10_000, 2048)
    assert led.reserved_cents == pytest.approx(held.cents)
    assert led.confirmed_cents == 0.0
    assert led.unresolved_cents == 0.0

    led.confirm(held, 10_000, 300)
    assert led.reserved_cents == pytest.approx(0.0)
    assert led.confirmed_cents == pytest.approx(cost_cents(OPUS, 10_000, 300))

    lost = led.reserve(10_000, 2048)
    led.mark_unresolved(lost, "timeout")
    assert led.reserved_cents == pytest.approx(0.0)
    assert led.unresolved_cents == pytest.approx(lost.cents)

    report = led.report()
    assert {"confirmed_cents", "reserved_cents", "unresolved_cents"} <= set(report)
    assert report["unresolved_reasons"] == ["timeout"]


def test_the_margin_is_applied_to_the_reservation():
    led = SpendLedger(cap_cents=10_000.0, model=OPUS)
    plain = led.reserve(10_000, 1000)
    led.confirm(plain, 10_000, 1000)
    with_margin = led.reserve(10_000, 1000, margin=0.25)
    assert with_margin.cents > plain.cents
    assert with_margin.input_tokens == 12_500


# --------------------------------------------------------------------------
# enforcement during execution
# --------------------------------------------------------------------------


def test_a_request_that_does_not_fit_is_refused_before_it_is_sent():
    led = SpendLedger(cap_cents=50.0, model=OPUS)
    refused = False
    for _ in range(50):
        try:
            led.reserve(20_000, 4_096)
        except BudgetExceeded:
            refused = True
            break
    assert refused
    assert led.refusals == 1


def test_unresolved_charges_shrink_the_spendable_balance_permanently():
    """An unresolved charge is money that may already be gone."""
    led = SpendLedger(cap_cents=100.0, model=OPUS)
    before = led.spendable_cents
    lost = led.reserve(5_000, 1_000)
    led.mark_unresolved(lost, "connection dropped after send")
    assert led.spendable_cents == pytest.approx(before - lost.cents)
    # and it never comes back
    led.release_unsent(lost)
    assert led.unresolved_cents == pytest.approx(lost.cents)


def test_only_a_provably_unsent_request_releases_its_hold():
    led = SpendLedger(cap_cents=100.0, model=OPUS)
    held = led.reserve(5_000, 1_000)
    led.release_unsent(held)
    assert led.spendable_cents == pytest.approx(100.0)
    assert led.unresolved_cents == 0.0


def test_reservations_bound_concurrent_requests():
    """The pilot runs sequentially, but the invariant must not depend on that."""
    led = SpendLedger(cap_cents=100.0, model=OPUS)
    per_call = cost_cents(OPUS, 10_000, 2_000)
    granted: list[float] = []
    refused: list[BudgetExceeded] = []

    def worker() -> None:
        try:
            granted.append(led.reserve(10_000, 2_000).cents)
        except BudgetExceeded as exc:
            refused.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == int(100.0 // per_call)
    assert len(refused) == 40 - len(granted)
    assert led.committed_cents <= led.cap_cents


def test_confirming_records_actual_usage_not_the_worst_case_estimate():
    led = SpendLedger(cap_cents=100.0, model=OPUS)
    held = led.reserve(10_000, 4_096)
    actual = led.confirm(held, 10_000, 120)
    assert actual < held.cents
    assert led.confirmed_cents == pytest.approx(actual)


# --------------------------------------------------------------------------
# persistence: a resumed run cannot reset its allowance
# --------------------------------------------------------------------------


def test_resuming_carries_prior_spend_forward(tmp_path):
    """A run restarting from zero would grant itself the whole cap again."""
    path = tmp_path / "ledger.json"
    first = SpendLedger(cap_cents=100.0, model=OPUS)
    held = first.reserve(10_000, 1_000)
    first.confirm(held, 10_000, 500)
    lost = first.reserve(10_000, 1_000)
    first.mark_unresolved(lost, "timeout")
    first.save(path)

    resumed = SpendLedger.resume(path, 100.0, OPUS)
    assert resumed.confirmed_cents == pytest.approx(first.confirmed_cents)
    assert resumed.unresolved_cents == pytest.approx(first.unresolved_cents)
    assert resumed.spendable_cents == pytest.approx(first.spendable_cents)
    assert resumed.unresolved_reasons == ["timeout"]


def test_an_in_flight_reservation_is_not_restored_as_spendable(tmp_path):
    """A request in flight when the process died has an unknown outcome."""
    path = tmp_path / "ledger.json"
    led = SpendLedger(cap_cents=100.0, model=OPUS)
    led.reserve(10_000, 1_000)  # never resolved; process "dies" here
    led.save(path)
    saved = json.loads(path.read_text())
    assert "reserved_cents" not in saved

    resumed = SpendLedger.resume(path, 100.0, OPUS)
    assert resumed.reserved_cents == 0.0


def test_resuming_refuses_a_changed_cap_or_model(tmp_path):
    path = tmp_path / "ledger.json"
    SpendLedger(cap_cents=100.0, model=OPUS).save(path)
    with pytest.raises(ValueError, match="capped at"):
        SpendLedger.resume(path, 500.0, OPUS)
    with pytest.raises(ValueError, match="refusing to resume"):
        SpendLedger.resume(path, 100.0, "claude-sonnet-5")


def test_a_missing_ledger_starts_clean(tmp_path):
    led = SpendLedger.resume(tmp_path / "absent.json", 100.0, OPUS)
    assert led.confirmed_cents == 0.0
    assert led.cap_cents == 100.0
