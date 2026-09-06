"""The inference deadline, verified with a delayed fake transport.

No model server and no model runtime: timeout behaviour is exercised by making
a fake transport sleep, which is both faster and more controllable than
spending real generation time to provoke a timeout.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from cerl.agents.local_client import LocalModelClient, LocalModelInfo, LocalRunnerUnavailable
from cerl.eval.latency import CANCELLATION_NOTE, Deadline, DeadlineExceeded


class SlowTransport:
    """Sleeps, then answers -- or raises as a bounded urllib timeout would."""

    def __init__(self, delay_s: float, *, fail_after: float | None = None) -> None:
        self.delay_s = delay_s
        self.fail_after = fail_after
        self.timeouts_seen: list[float | None] = []
        self.calls = 0

    def chat(self, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        self.calls += 1
        self.timeouts_seen.append(timeout_s)
        time.sleep(self.delay_s)
        if self.fail_after is not None and timeout_s is not None and timeout_s <= self.fail_after:
            raise LocalRunnerUnavailable("timed out")
        return {
            "message": {"content": "", "tool_calls": [
                {"function": {"name": "abstain", "arguments": {}}}]},
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5,
        }

    def version(self) -> str:
        return "0-fake"

    def tags(self) -> dict[str, Any]:
        return {"models": []}


def _client(transport: Any, deadline: Deadline) -> LocalModelClient:
    return LocalModelClient(
        transport=transport, deadline=deadline, info=LocalModelInfo(digest="d" * 64),
    )


def test_each_request_is_bounded_by_the_remaining_allowance():
    """A single slow call must not overrun the budget by its own timeout."""
    transport = SlowTransport(0.01)
    deadline = Deadline(budget_seconds=30.0)
    client = _client(transport, deadline)
    client.complete("sys", [], [])
    first = transport.timeouts_seen[0]
    assert first is not None
    assert first <= 30.0
    client.complete("sys", [], [])
    # the second request is bounded by strictly less than the first
    assert transport.timeouts_seen[1] < first


def test_no_request_starts_after_the_deadline():
    transport = SlowTransport(0.01)
    deadline = Deadline(budget_seconds=30.0, spent_seconds=30.0)
    with pytest.raises(DeadlineExceeded):
        _client(transport, deadline).complete("sys", [], [])
    assert transport.calls == 0, "a request was sent after the deadline"
    assert deadline.requests_refused == 1


def test_a_nearly_spent_budget_refuses_rather_than_starting_a_doomed_request():
    """Half a second buys nothing but a truncated response to throw away."""
    transport = SlowTransport(0.01)
    deadline = Deadline(budget_seconds=30.0, spent_seconds=29.5)
    with pytest.raises(DeadlineExceeded):
        _client(transport, deadline).complete("sys", [], [])
    assert transport.calls == 0


def test_time_spent_is_recorded_by_the_attempt_not_the_success():
    transport = SlowTransport(0.05)
    deadline = Deadline(budget_seconds=30.0)
    _client(transport, deadline).complete("sys", [], [])
    assert deadline.spent_seconds >= 0.05


def test_an_abandoned_request_is_counted_and_not_claimed_as_cancelled():
    """We cannot see whether the server stopped, so we do not claim it did."""
    transport = SlowTransport(0.02, fail_after=100.0)
    deadline = Deadline(budget_seconds=0.01)
    deadline.spent_seconds = 0.0
    # budget under the 1s minimum -> refused before sending
    with pytest.raises(DeadlineExceeded):
        _client(transport, deadline).complete("sys", [], [])
    report = deadline.report()
    assert report["cancellation_note"] == CANCELLATION_NOTE
    assert "NOT confirmed" in str(report["cancellation_note"])


def test_a_mid_flight_timeout_at_the_deadline_is_recorded_as_abandoned():
    """The request was allowed to start, then ran the budget out."""
    transport = SlowTransport(1.1, fail_after=100.0)
    deadline = Deadline(budget_seconds=1.05)
    with pytest.raises(DeadlineExceeded, match="NOT confirmed"):
        _client(transport, deadline).complete("sys", [], [])
    assert transport.calls == 1
    assert deadline.requests_abandoned == 1
    assert deadline.expired


def test_a_timeout_before_the_deadline_stays_an_ordinary_runner_failure():
    """Not every timeout is a budget event; conflating them would hide outages."""
    transport = SlowTransport(0.02, fail_after=100.0)
    deadline = Deadline(budget_seconds=60.0)
    with pytest.raises(LocalRunnerUnavailable):
        _client(transport, deadline).complete("sys", [], [])
    assert deadline.requests_abandoned == 0


def test_the_report_separates_refused_from_abandoned():
    """Different failures: one never left, one may have consumed server time."""
    deadline = Deadline(budget_seconds=10.0)
    deadline.record(4.0)
    report = deadline.report()
    assert report["spent_seconds"] == 4.0
    assert report["remaining_seconds"] == 6.0
    assert report["requests_refused_after_deadline"] == 0
    assert report["requests_abandoned_mid_flight"] == 0


def test_the_deadline_never_touches_simulated_time():
    """It bounds waiting on a server; logical time advances only by tick_cost."""
    from cerl.actions import ActionKind
    from cerl.tools import tick_cost

    before = tick_cost(ActionKind.TICKETS_GET)
    deadline = Deadline(budget_seconds=1.0)
    deadline.record(0.9)
    assert tick_cost(ActionKind.TICKETS_GET) == before
    assert tick_cost(ActionKind.MALFORMED) == 0


def test_a_client_without_a_deadline_is_unbounded_as_before():
    transport = SlowTransport(0.01)
    client = LocalModelClient(transport=transport, info=LocalModelInfo(digest="d" * 64))
    client.complete("sys", [], [])
    assert transport.timeouts_seen == [None]


def test_a_socket_timeout_is_classified_not_leaked_as_a_bare_error():
    """Regression: the real run leaked a raw ``TimeoutError``.

    ``urlopen`` raises ``TimeoutError`` for a bounded request that runs out of
    time, and ``TimeoutError`` is not a ``URLError``. Catching only URLError let
    it escape the deadline accounting entirely, so an external timeout was
    recorded as a generic interruption instead of what it was.
    """
    import urllib.request

    from cerl.agents.local_client import OllamaTransport

    transport = OllamaTransport(host="http://127.0.0.1:11434")

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("timed out")

    original = urllib.request.urlopen
    urllib.request.urlopen = _raise  # type: ignore[assignment]
    try:
        with pytest.raises(LocalRunnerUnavailable, match="No remote provider"):
            transport.chat({"model": "x"}, 1.0)
    finally:
        urllib.request.urlopen = original  # type: ignore[assignment]


def test_a_deadline_timeout_is_reported_as_an_external_timeout():
    """The termination line must say what happened, not merely that it stopped."""
    from cerl.eval.local_run import _termination

    class _Client:
        deadline = Deadline(budget_seconds=1800.0, spent_seconds=1650.8)

    class _Manifest:
        episodes: tuple[object, ...] = ()

    class _Result:
        manifest = _Manifest()
        interrupted = ("scenario-x: TimeoutError: timed out",)

    lines = _termination(_Result(), _Client())  # type: ignore[arg-type]
    assert lines
    assert "EXTERNAL TIMEOUT at the inference deadline" in lines[0]
    assert "1650.8" in lines[0]
    assert "NOT confirmed" in lines[0]


def test_every_external_stop_preserves_partial_work():
    """One category, one behaviour.

    A deadline, a dead server, an exhausted budget and an ambiguous request all
    stop the run from outside the simulation. Each must preserve the episodes
    already executed; none of them may be confused with a defect in the
    evidence, which must still surface.
    """
    from cerl.agents.budget import BudgetExceeded
    from cerl.agents.model_client import AmbiguousRequestOutcome, TranscriptCacheMiss
    from cerl.core import ExternalInterruption

    for external in (
        DeadlineExceeded,
        LocalRunnerUnavailable,
        BudgetExceeded,
        AmbiguousRequestOutcome,
    ):
        assert issubclass(external, ExternalInterruption), external

    # A cache miss is evidence being wrong, not the world interrupting us.
    assert not issubclass(TranscriptCacheMiss, ExternalInterruption)
