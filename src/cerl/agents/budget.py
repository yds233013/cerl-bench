"""Spend control for live model evaluation.

Requiring a positive budget when the client is constructed is not spending
control: it checks once, before anything has been spent, and then never again.
A run could exceed it on the second request and keep going.

This ledger enforces the cap *during* execution. Every request is checked
against the remaining budget **before** it is issued, using a conservative
worst-case estimate (the full `max_tokens` billed as output), and the actual
cost is recorded from the response's own usage afterwards. A request that could
push the total past the cap is refused rather than attempted.

Three details that matter more than the arithmetic:

* **Retries are charged.** A retried request is a second request; the ledger
  reserves for it separately. Retry budgets that only count "logical" calls are
  how a run bills three times what it planned.
* **Outstanding requests are reserved, not hoped for.** The reservation is
  deducted before the call and reconciled after, so concurrent or in-flight
  requests cannot each individually pass a check that they jointly fail.
* **The estimate is worst-case.** Output tokens are unknown until the response
  arrives, so the pre-flight check assumes the full `max_tokens`. The ledger
  therefore stops early rather than late.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

#: US dollars per million tokens, Anthropic first-party API, Claude Opus 5.
#: Cached from the pricing table on 2026-09-05; re-check before a real run.
OPUS_5_INPUT_PER_MTOK = 5.00
OPUS_5_OUTPUT_PER_MTOK = 25.00

PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (OPUS_5_INPUT_PER_MTOK, OPUS_5_OUTPUT_PER_MTOK),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class BudgetExceeded(RuntimeError):
    """A request was refused because it could push the run past its cap."""


def cost_cents(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one request in US cents."""
    if model not in PRICING:
        raise KeyError(f"no cached pricing for {model!r}; add it before running")
    input_rate, output_rate = PRICING[model]
    dollars = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return dollars * 100.0


@dataclass
class BudgetLedger:
    """Tracks spend against a cap, reserving before each request.

    Thread-safe, because "outstanding requests are reserved" is only true if two
    threads cannot both pass the check on the same remaining balance.
    """

    cap_cents: float
    model: str
    spent_cents: float = 0.0
    reserved_cents: float = 0.0
    requests: int = 0
    refusals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def committed_cents(self) -> float:
        """Everything spent plus everything currently reserved in flight."""
        return self.spent_cents + self.reserved_cents

    @property
    def remaining_cents(self) -> float:
        return max(0.0, self.cap_cents - self.committed_cents)

    def reserve(self, estimated_input_tokens: int, max_output_tokens: int) -> float:
        """Reserve worst-case cost for one request, or refuse it.

        Called before every request including every retry. Returns the amount
        reserved, which must be passed back to :meth:`settle`.
        """
        estimate = cost_cents(self.model, estimated_input_tokens, max_output_tokens)
        with self._lock:
            if self.spent_cents + self.reserved_cents + estimate > self.cap_cents:
                self.refusals += 1
                raise BudgetExceeded(
                    f"refusing request: worst-case {estimate:.3f}c on top of "
                    f"{self.committed_cents:.3f}c would exceed the "
                    f"{self.cap_cents:.2f}c cap",
                )
            self.reserved_cents += estimate
            return estimate

    def settle(self, reserved: float, input_tokens: int, output_tokens: int) -> float:
        """Record what a completed request actually cost and release its reservation."""
        actual = cost_cents(self.model, input_tokens, output_tokens)
        with self._lock:
            self.reserved_cents = max(0.0, self.reserved_cents - reserved)
            self.spent_cents += actual
            self.requests += 1
            return actual

    def release(self, reserved: float) -> None:
        """Release a reservation for a request that never completed."""
        with self._lock:
            self.reserved_cents = max(0.0, self.reserved_cents - reserved)

    def summary(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "cap_cents": self.cap_cents,
            "spent_cents": round(self.spent_cents, 4),
            "reserved_cents": round(self.reserved_cents, 4),
            "remaining_cents": round(self.remaining_cents, 4),
            "requests": self.requests,
            "refusals": self.refusals,
        }


def estimate_input_tokens(text_length_chars: int) -> int:
    """A deliberately pessimistic characters-to-tokens estimate.

    Roughly 3.2 characters per token rather than the usual ~4, so the pre-flight
    check over-reserves. Under-estimating input is the one error that lets a run
    exceed its cap, so the bias is chosen on purpose. Replace with
    ``client.messages.count_tokens`` before any run where the margin matters.
    """
    return int(text_length_chars / 3.2) + 1
