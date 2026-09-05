"""The spend ledger must bound cost *during* a run, not just at construction."""

from __future__ import annotations

import threading

import pytest

from cerl.agents.budget import (
    BudgetExceeded,
    BudgetLedger,
    cost_cents,
    estimate_input_tokens,
)


def test_pricing_arithmetic_is_right():
    # 1M input at $5 = $5 = 500c; 1M output at $25 = 2500c.
    assert cost_cents("claude-opus-5", 1_000_000, 0) == pytest.approx(500.0)
    assert cost_cents("claude-opus-5", 0, 1_000_000) == pytest.approx(2500.0)
    assert cost_cents("claude-opus-5", 20_000, 1_000) == pytest.approx(12.5)


def test_unknown_model_is_refused_not_guessed():
    """A missing price must stop the run, never default to zero.

    Defaulting would make an unpriced model appear free and uncapped.
    """
    with pytest.raises(KeyError):
        cost_cents("some-future-model", 1, 1)


def test_a_request_that_would_exceed_the_cap_is_refused_before_it_is_sent():
    led = BudgetLedger(cap_cents=50.0, model="claude-opus-5")
    r = led.reserve(20_000, 4_096)
    led.settle(r, 20_000, 900)
    assert led.spent_cents == pytest.approx(cost_cents("claude-opus-5", 20_000, 900))
    refused = False
    for _ in range(50):
        try:
            led.reserve(20_000, 4_096)
        except BudgetExceeded:
            refused = True
            break
    assert refused, "the ledger never refused a request despite exceeding the cap"
    assert led.refusals >= 1


def test_reservations_bound_concurrent_requests():
    """Outstanding requests are reserved, so they cannot jointly overrun.

    Without reservation, N in-flight requests each check against the same
    unspent balance and all pass. This is the failure the ledger exists for.
    """
    led = BudgetLedger(cap_cents=100.0, model="claude-opus-5")
    per_call = cost_cents("claude-opus-5", 10_000, 2_000)
    granted: list[float] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            granted.append(led.reserve(10_000, 2_000))
        except BudgetExceeded as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == int(100.0 // per_call)
    assert len(errors) == 40 - len(granted)
    assert led.reserved_cents <= led.cap_cents


def test_every_retry_is_charged_separately():
    """A retry is another request. Reserving once per logical call underbills."""
    led = BudgetLedger(cap_cents=200.0, model="claude-opus-5")
    # one logical call, three attempts: two failures released, one settled
    for _ in range(2):
        r = led.reserve(5_000, 1_000)
        led.release(r)  # failed before producing usage
    r = led.reserve(5_000, 1_000)
    led.settle(r, 5_000, 400)
    assert led.requests == 1
    assert led.reserved_cents == pytest.approx(0.0)
    # failures released their reservation but the cap check saw them at the time
    assert led.spent_cents == pytest.approx(cost_cents("claude-opus-5", 5_000, 400))


def test_a_failed_request_releases_its_reservation():
    led = BudgetLedger(cap_cents=30.0, model="claude-opus-5")
    r = led.reserve(10_000, 1_000)
    assert led.remaining_cents < 30.0
    led.release(r)
    assert led.remaining_cents == pytest.approx(30.0)


def test_settling_records_actual_not_estimated_cost():
    """The estimate is worst-case; the ledger must not keep charging it."""
    led = BudgetLedger(cap_cents=100.0, model="claude-opus-5")
    r = led.reserve(10_000, 4_096)
    actual = led.settle(r, 10_000, 120)
    assert actual < r
    assert led.spent_cents == pytest.approx(actual)
    assert led.reserved_cents == pytest.approx(0.0)


def test_input_estimate_is_biased_high():
    """Under-estimating input is the one error that breaks the cap."""
    text = "x" * 4000
    # a ~4 chars/token estimate would be 1000; ours must exceed it
    assert estimate_input_tokens(len(text)) > len(text) / 4
