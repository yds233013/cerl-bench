"""Spend accounting for live model evaluation.

An earlier version of this module claimed a character-based token estimate made
overspending impossible. **That claim was wrong and has been withdrawn.** A
heuristic biased high is not a guarantee: it can be wrong in the other direction
on unusual input, and it says nothing at all about requests whose outcome we
never learn. What this module provides is an *operational limit* with its
assumptions written down, not a dollar guarantee. ``docs/budget-accounting.md``
states the residual exposure.

Four quantities are tracked separately, because collapsing them is how an
accounting system lies:

===============  ==========================================================
``confirmed``    Billed usage read back from a provider response. Fact.
``reserved``     Held for a request that is in flight right now. Released on
                 a definite outcome -- never on an ambiguous one.
``unresolved``   Held **permanently** for a request that may have reached the
                 provider but whose usage we never learned: a timeout, a
                 dropped connection after send, a response without usage. We
                 may well have been billed, so this is never given back.
``estimated``    A projection for a request not yet sent. Not money.
===============  ==========================================================

The spendable balance is ``cap - (confirmed + reserved + unresolved)``. A
request is refused when its worst-case estimate does not fit in that balance.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------

#: US dollars per million tokens. Cached from the published first-party API
#: pricing on 2026-09-05. Re-check before any run: a stale table under-reports.
#: Tuple order: (input, output, cache_write_5m, cache_read).
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}

PRICING_AS_OF = "2026-09-05"


class BudgetExceeded(RuntimeError):
    """A request was refused because it did not fit in the spendable balance."""


class UnpricedModel(KeyError):
    """No cached price for a model.

    Raised rather than defaulting to zero: an unpriced model that costs nothing
    on paper would run entirely uncapped.
    """


def _rates(model: str) -> tuple[float, float, float, float]:
    if model not in PRICING:
        raise UnpricedModel(
            f"no cached pricing for {model!r} (table as of {PRICING_AS_OF}); "
            f"add it before running rather than treating it as free",
        )
    return PRICING[model]


def cost_cents(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Cost of one request in US cents, including cache pricing modifiers.

    Thinking tokens are **not** a separate line: the provider bills them as
    output, and they are drawn from the same ``max_tokens`` allowance, so a
    worst-case reservation of ``max_tokens`` output already covers them.
    """
    in_rate, out_rate, write_rate, read_rate = _rates(model)
    dollars = (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_write_tokens * write_rate
        + cache_read_tokens * read_rate
    ) / 1_000_000
    return dollars * 100.0


# --------------------------------------------------------------------------
# token counting
# --------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Counts the input tokens of a complete request.

    "Complete" is load-bearing: the count must cover the system prompt, every
    message in the conversation so far, and the tool schemas. Counting only the
    newest message under-reserves by the size of the whole transcript, which on
    a 40-step episode is the overwhelming majority of the bill.
    """

    name: str
    #: Fractional allowance added to this counter's result before reserving.
    uncertainty_margin: float

    def count(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int: ...


#: Characters per token assumed by the heuristic. Deliberately below the usual
#: ~4 so the estimate leans high, but see the class docstring: leaning is not
#: bounding.
HEURISTIC_CHARS_PER_TOKEN = 3.2

#: Added on top of the heuristic count. 25% is a working allowance, not a proof.
HEURISTIC_MARGIN = 0.25

#: Added on top of a provider count. The provider's own counter is exact for
#: input, but the request it prices is not byte-identical to the one we send
#: (server-side additions), so a small allowance remains.
API_MARGIN = 0.05


class HeuristicTokenCounter:
    """Character-based fallback. Approximate, and labelled as such.

    Used when the provider's counting endpoint is unavailable. Its error is not
    bounded in either direction -- dense JSON tool schemas tokenize very
    differently from prose -- so a run using this counter has a wider residual
    exposure, recorded in the ledger and reported at the end.
    """

    name = "heuristic"
    uncertainty_margin = HEURISTIC_MARGIN

    def count(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        payload = json.dumps(
            {"system": system, "messages": messages, "tools": tools},
            sort_keys=True,
            ensure_ascii=False,
        )
        return int(len(payload) / HEURISTIC_CHARS_PER_TOKEN) + 1


class ApiTokenCounter:
    """Uses the provider's own ``messages.count_tokens`` endpoint.

    Exact for input, which is the quantity that dominates a long agentic
    episode. Counting is itself an API call, but it is not billed as inference;
    it is still a network round trip per request, which is part of why this
    pilot runs sequentially.
    """

    name = "provider-count_tokens"
    uncertainty_margin = API_MARGIN

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def count(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        result = self._client.messages.count_tokens(
            model=self._model,
            system=system,
            messages=messages,
            tools=tools,
        )
        return int(result.input_tokens)


class MockTokenCounter:
    """Deterministic counter for offline tests. Never contacts a provider."""

    name = "mock"
    uncertainty_margin = 0.0

    def __init__(self, tokens_per_message: int = 100, base: int = 500) -> None:
        self._per_message = tokens_per_message
        self._base = base
        self.calls = 0

    def count(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        self.calls += 1
        # Grows with the transcript, like a real counter, so a test that
        # under-reserves the history fails here too.
        return (
            self._base
            + len(system) // 100
            + len(messages) * self._per_message
            + len(tools)
        )


def resolve_counter(client: Any, model: str) -> TokenCounter:
    """Prefer the provider's counter; fall back to the heuristic, and say so."""
    if client is not None and hasattr(getattr(client, "messages", None), "count_tokens"):
        return ApiTokenCounter(client, model)
    return HeuristicTokenCounter()


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reservation:
    """A hold placed before a request is sent."""

    cents: float
    input_tokens: int
    max_output_tokens: int


@dataclass
class SpendLedger:
    """Meters spend against a cap, reserving before every request.

    Thread-safe, though this pilot issues requests **sequentially** on purpose:
    with one request outstanding at a time, the balance at each decision point
    is unambiguous and the accounting is auditable line by line. The lock is
    kept so the invariant does not depend on that policy holding forever.
    """

    cap_cents: float
    model: str
    counter_name: str = "unknown"
    confirmed_cents: float = 0.0
    reserved_cents: float = 0.0
    #: Never decreases. Requests that may have been billed but whose usage we
    #: never saw.
    unresolved_cents: float = 0.0
    requests_confirmed: int = 0
    requests_unresolved: int = 0
    refusals: int = 0
    #: Why each unresolved charge is unresolved, so the exposure is explicable.
    unresolved_reasons: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- balances ---------------------------------------------------------

    @property
    def committed_cents(self) -> float:
        """Everything that is spent, held, or possibly spent."""
        return self.confirmed_cents + self.reserved_cents + self.unresolved_cents

    @property
    def spendable_cents(self) -> float:
        return max(0.0, self.cap_cents - self.committed_cents)

    # -- the request lifecycle -------------------------------------------

    def reserve(
        self,
        input_tokens: int,
        max_output_tokens: int,
        *,
        margin: float = 0.0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Reservation:
        """Hold the worst-case cost of one request, or refuse it.

        Called before **every** request, including every retry. Output is
        reserved at the full ``max_output_tokens`` because actual output --
        thinking included -- is unknown until the response arrives.
        """
        billable_input = int(input_tokens * (1.0 + margin))
        estimate = cost_cents(
            self.model,
            billable_input,
            max_output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        with self._lock:
            if self.committed_cents + estimate > self.cap_cents:
                self.refusals += 1
                raise BudgetExceeded(
                    f"refusing request: worst-case {estimate:.3f}c does not fit in "
                    f"the {self.spendable_cents:.3f}c spendable balance "
                    f"(cap {self.cap_cents:.2f}c; confirmed {self.confirmed_cents:.3f}c, "
                    f"reserved {self.reserved_cents:.3f}c, "
                    f"unresolved {self.unresolved_cents:.3f}c)",
                )
            self.reserved_cents += estimate
        return Reservation(estimate, billable_input, max_output_tokens)

    def confirm(
        self,
        reservation: Reservation,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Record real billed usage and release the hold."""
        actual = cost_cents(
            self.model,
            input_tokens,
            output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        with self._lock:
            self.reserved_cents = max(0.0, self.reserved_cents - reservation.cents)
            self.confirmed_cents += actual
            self.requests_confirmed += 1
        return actual

    def mark_unresolved(self, reservation: Reservation, reason: str) -> None:
        """Convert a hold into a permanent unresolved charge.

        For any request that **may** have reached the provider: a timeout, a
        connection dropped after send, a response that arrived without usage.
        We were possibly billed, and possibly billed the full amount, so the
        hold is kept against the cap forever rather than returned. Treating an
        ambiguous outcome as free is the single easiest way to overspend.
        """
        with self._lock:
            self.reserved_cents = max(0.0, self.reserved_cents - reservation.cents)
            self.unresolved_cents += reservation.cents
            self.requests_unresolved += 1
        self.unresolved_reasons.append(reason)

    def release_unsent(self, reservation: Reservation) -> None:
        """Release a hold for a request that provably never reached the provider.

        Only for failures raised before transmission -- a refusal by our own
        code, a request we chose not to send. Anything that touched the network
        goes to :meth:`mark_unresolved` instead.
        """
        with self._lock:
            self.reserved_cents = max(0.0, self.reserved_cents - reservation.cents)

    # -- persistence ------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Serialisable accounting state, for resuming without forgetting."""
        return {
            "cap_cents": self.cap_cents,
            "model": self.model,
            "counter_name": self.counter_name,
            "confirmed_cents": self.confirmed_cents,
            "unresolved_cents": self.unresolved_cents,
            "requests_confirmed": self.requests_confirmed,
            "requests_unresolved": self.requests_unresolved,
            "unresolved_reasons": list(self.unresolved_reasons),
            "pricing_as_of": PRICING_AS_OF,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def resume(cls, path: Path, cap_cents: float, model: str) -> SpendLedger:
        """Reload prior spend so a resumed run continues the same allowance.

        A resumed run that started from zero would grant itself the whole cap
        again, which is how "a $50 cap" becomes $50 per attempt. Reserved
        amounts are deliberately *not* restored as reserved: a request in flight
        when the process died has an unknown outcome, so it is already recorded
        as unresolved and stays charged.
        """
        if not path.exists():
            return cls(cap_cents=cap_cents, model=model)
        prior = json.loads(path.read_text())
        if prior.get("model") != model:
            raise ValueError(
                f"ledger at {path} is for model {prior.get('model')!r}, not {model!r}; "
                f"refusing to resume a different model's allowance",
            )
        if float(prior.get("cap_cents", cap_cents)) != cap_cents:
            raise ValueError(
                f"ledger at {path} was capped at {prior.get('cap_cents')}c, not "
                f"{cap_cents}c; a changed cap must be an explicit new run",
            )
        ledger = cls(
            cap_cents=cap_cents,
            model=model,
            counter_name=str(prior.get("counter_name", "unknown")),
            confirmed_cents=float(prior.get("confirmed_cents", 0.0)),
            unresolved_cents=float(prior.get("unresolved_cents", 0.0)),
            requests_confirmed=int(prior.get("requests_confirmed", 0)),
            requests_unresolved=int(prior.get("requests_unresolved", 0)),
        )
        ledger.unresolved_reasons.extend(prior.get("unresolved_reasons", []))
        return ledger

    # -- reporting --------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Four quantities, kept apart, plus what is not known."""
        return {
            "model": self.model,
            "cap_cents": round(self.cap_cents, 4),
            "confirmed_cents": round(self.confirmed_cents, 4),
            "reserved_cents": round(self.reserved_cents, 4),
            "unresolved_cents": round(self.unresolved_cents, 4),
            "spendable_cents": round(self.spendable_cents, 4),
            "requests_confirmed": self.requests_confirmed,
            "requests_unresolved": self.requests_unresolved,
            "refusals": self.refusals,
            "token_counter": self.counter_name,
            "pricing_as_of": PRICING_AS_OF,
            "unresolved_reasons": list(self.unresolved_reasons),
        }

