"""Crash recovery at the real persistence and network boundaries.

These tests do not simulate a crash by calling a "crash" method. They interrupt
the process at the two places where money is actually at risk:

* **inside ``transport.create``** -- the request is on the wire, or may be, and
  the outcome record has not been written;
* **between the journal write and the ledger snapshot** -- the durable record
  and the in-memory position disagree.

Recovery is then performed from the files on disk alone, exactly as a restarted
process would see them.
"""

from __future__ import annotations

import json

import pytest

from cerl.agents.budget import (
    MockTokenCounter,
    SpendJournal,
    SpendLedger,
    cost_cents,
)
from cerl.agents.model_client import AnthropicClient
from cerl.agents.synthetic_transport import SyntheticTransport, tool_turn

OPUS = "claude-opus-5"
TOOLS = [{"name": "tickets__get", "input_schema": {"type": "object"}}]


class ProcessDied(BaseException):
    """Interrupts the process the way a SIGKILL would, mid-request.

    Deliberately a ``BaseException``: it must not be caught by the client's
    ``except Exception`` retry handler, because a real crash gets no chance to
    record an outcome. If it were catchable, this test would exercise the
    orderly-failure path instead of the crash path.
    """


def _ledger(tmp_path, cap=10_000.0):
    snapshot = tmp_path / "ledger.json"
    journal = tmp_path / "ledger.jsonl"
    return snapshot, journal, SpendLedger(
        cap_cents=cap, model=OPUS, journal=SpendJournal(journal),
    )


def _client(ledger, transport):
    return AnthropicClient(
        model=OPUS, max_tokens=2048, ledger=ledger, transport=transport,
        counter=MockTokenCounter(), require_authorization=False,
    )


# --------------------------------------------------------------------------
# the boundary that matters: dying inside the send
# --------------------------------------------------------------------------


def test_a_crash_inside_the_send_leaves_a_charged_orphan(tmp_path):
    """The request may have reached the provider. It must not be forgotten."""
    snapshot, journal, ledger = _ledger(tmp_path)

    def die(_index):
        raise ProcessDied

    transport = SyntheticTransport([tool_turn("tickets__get", {}), die])
    client = _client(ledger, transport)

    client.complete("sys", [], TOOLS)          # request 1 completes
    ledger.save(snapshot)
    with pytest.raises(ProcessDied):           # request 2 dies on the wire
        client.complete("sys", [], TOOLS)

    # A fresh process sees only the files.
    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal)
    assert recovered.requests_confirmed == 1
    assert recovered.requests_unresolved == 1
    assert recovered.unresolved_cents > 0.0
    assert len(recovered.orphaned_requests) == 1


def test_confirmed_usage_survives_a_crash_later_in_the_same_episode(tmp_path):
    """The loss window this journal exists to close.

    A snapshot written once per episode is stale by every request since. Twelve
    confirmed requests followed by a crash used to recover as zero.
    """
    snapshot, journal, ledger = _ledger(tmp_path)
    ledger.save(snapshot)  # the last snapshot, at episode start

    transport = SyntheticTransport(default=tool_turn("tickets__get", {}))
    client = _client(ledger, transport)
    for _ in range(12):
        client.complete("sys", [], TOOLS)
    spent = ledger.confirmed_cents
    assert spent > 0.0

    # crash before any further snapshot
    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal)
    assert recovered.confirmed_cents == pytest.approx(spent)
    assert recovered.requests_confirmed == 12


def test_the_allowance_is_not_reset_by_recovery(tmp_path):
    snapshot, journal, ledger = _ledger(tmp_path, cap=200.0)
    ledger.save(snapshot)
    transport = SyntheticTransport(default=tool_turn("tickets__get", {}))
    client = _client(ledger, transport)
    for _ in range(5):
        client.complete("sys", [], TOOLS)

    recovered = SpendLedger.resume(snapshot, 200.0, OPUS, journal_path=journal)
    assert recovered.spendable_cents < 200.0
    assert recovered.spendable_cents == pytest.approx(ledger.spendable_cents)


def test_recovery_never_replays_an_uncertain_request(tmp_path):
    """Restarting restores an accounting position, not a request queue.

    Auto-replaying an orphan would double-bill a request that may have
    succeeded. Whether to retry is a person's decision.
    """
    snapshot, journal, ledger = _ledger(tmp_path)

    def die(_index):
        raise ProcessDied

    transport = SyntheticTransport([die])
    client = _client(ledger, transport)
    with pytest.raises(ProcessDied):
        client.complete("sys", [], TOOLS)
    ledger.save(snapshot)

    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal)
    fresh_transport = SyntheticTransport(default=tool_turn("tickets__get", {}))
    _client(recovered, fresh_transport)
    # Constructing a client over a recovered ledger issues nothing at all.
    assert fresh_transport.requests == []
    assert recovered.orphaned_requests


# --------------------------------------------------------------------------
# journal durability
# --------------------------------------------------------------------------


def test_the_reservation_is_on_disk_before_the_request_is_sent(tmp_path):
    """Ordering is the guarantee; assert it at the transport boundary."""
    _, journal_path, ledger = _ledger(tmp_path)
    seen: list[int] = []

    def observe(_index):
        # Called during create(): the reservation must already be durable.
        seen.append(len(journal_path.read_text().splitlines()))
        return tool_turn("tickets__get", {})

    client = _client(ledger, SyntheticTransport([observe]))
    client.complete("sys", [], TOOLS)
    assert seen == [1], "no reservation record existed when the request was sent"


def test_a_truncated_final_record_does_not_break_recovery(tmp_path):
    """A crash mid-write leaves a partial line. Everything before it is intact."""
    snapshot, journal_path, ledger = _ledger(tmp_path)
    ledger.save(snapshot)
    client = _client(ledger, SyntheticTransport(default=tool_turn("tickets__get", {})))
    for _ in range(3):
        client.complete("sys", [], TOOLS)

    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "reserved", "request_id": "099", "cen')

    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal_path)
    assert recovered.requests_confirmed == 3
    assert recovered.confirmed_cents == pytest.approx(ledger.confirmed_cents)


def test_a_request_that_never_left_the_process_is_not_charged(tmp_path):
    """`release_unsent` records a zero resolution so recovery sees no orphan."""
    snapshot, journal_path, ledger = _ledger(tmp_path)
    held = ledger.reserve(1000, 100)
    ledger.release_unsent(held)
    ledger.save(snapshot)

    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal_path)
    assert recovered.unresolved_cents == 0.0
    assert recovered.orphaned_requests == []


def test_the_journal_wins_over_a_stale_snapshot(tmp_path):
    """Two records disagreeing is the normal post-crash state, not an error."""
    snapshot, journal_path, ledger = _ledger(tmp_path)
    ledger.save(snapshot)  # snapshot says zero
    client = _client(ledger, SyntheticTransport(default=tool_turn("tickets__get", {})))
    client.complete("sys", [], TOOLS)

    assert json.loads(snapshot.read_text())["confirmed_cents"] == 0.0
    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal_path)
    assert recovered.confirmed_cents > 0.0


def test_resumption_does_not_reuse_request_ids(tmp_path):
    """A reused id would silently resolve an older record."""
    snapshot, journal_path, ledger = _ledger(tmp_path)
    ledger.save(snapshot)
    client = _client(ledger, SyntheticTransport(default=tool_turn("tickets__get", {})))
    for _ in range(3):
        client.complete("sys", [], TOOLS)

    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal_path)
    new_client = _client(
        recovered, SyntheticTransport(default=tool_turn("tickets__get", {})),
    )
    new_client.complete("sys", [], TOOLS)
    ids = [
        json.loads(line)["request_id"]
        for line in journal_path.read_text().splitlines()
        if line.strip() and json.loads(line)["event"] == "reserved"
    ]
    assert len(ids) == len(set(ids)), ids


# --------------------------------------------------------------------------
# the model and cap restrictions still hold across recovery
# --------------------------------------------------------------------------


def test_recovery_still_refuses_a_changed_cap_or_model(tmp_path):
    snapshot, journal_path, ledger = _ledger(tmp_path)
    ledger.save(snapshot)
    with pytest.raises(ValueError, match="capped at"):
        SpendLedger.resume(snapshot, 999.0, OPUS, journal_path=journal_path)
    with pytest.raises(ValueError, match="refusing to resume"):
        SpendLedger.resume(snapshot, 10_000.0, "claude-sonnet-5", journal_path=journal_path)


def test_a_recovered_ledger_still_refuses_requests_past_the_cap(tmp_path):
    """Recovery restores the limit, not just the number."""
    snapshot, journal_path, ledger = _ledger(tmp_path, cap=40.0)
    ledger.save(snapshot)
    client = _client(ledger, SyntheticTransport(default=tool_turn("tickets__get", {})))
    client.complete("sys", [], TOOLS)

    recovered = SpendLedger.resume(snapshot, 40.0, OPUS, journal_path=journal_path)
    from cerl.agents.budget import BudgetExceeded

    refused = False
    for _ in range(50):
        try:
            recovered.reserve(20_000, 4_096)
        except BudgetExceeded:
            refused = True
            break
    assert refused, "a recovered ledger stopped enforcing its cap"
    assert recovered.committed_cents <= 40.0


def test_pricing_of_a_recovered_position_is_unchanged(tmp_path):
    snapshot, journal_path, ledger = _ledger(tmp_path)
    ledger.save(snapshot)
    client = _client(ledger, SyntheticTransport(default=tool_turn(
        "tickets__get", {}, input_tokens=1000, output_tokens=100)))
    client.complete("sys", [], TOOLS)
    recovered = SpendLedger.resume(snapshot, 10_000.0, OPUS, journal_path=journal_path)
    assert recovered.confirmed_cents == pytest.approx(cost_cents(OPUS, 1000, 100))
