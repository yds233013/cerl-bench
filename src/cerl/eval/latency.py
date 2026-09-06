"""Measuring how long an *external* process took. The one wall-clock site.

CLAUDE.md rule 1 bans the wall clock from ``src/cerl/`` because simulated time
must be `LogicalClock` ticks: a scenario whose behaviour depends on how fast the
machine ran it is not reproducible, and that is the whole product.

This module is the single, enumerated exception, and it does not weaken that
rule. What it measures is **latency of a process outside the simulation** -- how
many seconds a local model server spent generating. That number:

* never enters ``WorldState``, a ``StateDiff``, a ``TraceEntry`` or a ``Verdict``;
* never influences an action, an observation, or a scored outcome;
* is reported as run *provenance*, in the same category as token counts.

An episode replays identically whether it originally took two seconds or two
hours, which is exactly the property rule 1 protects.

``tests/determinism`` enumerates this file by path, so a wall-clock call
appearing anywhere else in ``src/cerl/`` still fails the gate. Adding a second
exempt module requires editing that list deliberately.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from cerl.core import ExternalInterruption


@dataclass
class Elapsed:
    """Seconds spent inside a block, filled in when the block exits."""

    seconds: float = 0.0


@contextmanager
def measure() -> Iterator[Elapsed]:
    """Time an external call.

    Use for a network or subprocess round trip. Never for anything the
    environment's behaviour depends on.
    """
    result = Elapsed()
    started = time.monotonic()
    try:
        yield result
    finally:
        result.seconds = time.monotonic() - started


@dataclass
class Stopwatch:
    """Accumulates external latency across many calls."""

    total: float = 0.0
    samples: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.total += seconds
        self.samples.append(seconds)


class DeadlineExceeded(ExternalInterruption):
    """The wall-clock budget for external calls is spent."""


@dataclass
class Deadline:
    """A wall-clock budget for calls to an external service.

    Measured outside the simulator, and deliberately so: it bounds how long we
    are willing to wait on a model server, which is not simulated time and must
    never influence the environment. Logical time still advances only by
    ``tick_cost``.

    Two properties matter for honest accounting:

    * **No request starts after the deadline.** ``check`` raises rather than
      letting one more call slip past.
    * **A started request is bounded by what remains**, so a single slow call
      cannot overrun the budget by its own timeout.
    """

    budget_seconds: float
    spent_seconds: float = 0.0
    requests_refused: int = 0
    #: Requests abandoned when their bounded timeout expired. Whether the server
    #: kept generating is *not* knowable from here -- see ``cancellation_note``.
    requests_abandoned: int = 0

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget_seconds - self.spent_seconds)

    @property
    def expired(self) -> bool:
        return self.remaining_seconds <= 0.0

    def check(self, minimum_seconds: float = 1.0) -> float:
        """Remaining allowance, or raise if there is not enough left to try.

        Starting a request that cannot plausibly finish spends budget for
        nothing and produces a truncated response we would have to discard.
        """
        if self.remaining_seconds < minimum_seconds:
            self.requests_refused += 1
            raise DeadlineExceeded(
                f"inference deadline reached: {self.spent_seconds:.0f}s of "
                f"{self.budget_seconds:.0f}s spent. No further request started.",
            )
        return self.remaining_seconds

    def record(self, seconds: float) -> None:
        self.spent_seconds += seconds

    def abandon(self) -> None:
        self.requests_abandoned += 1

    def report(self) -> dict[str, float | int | str]:
        return {
            "budget_seconds": round(self.budget_seconds, 1),
            "spent_seconds": round(self.spent_seconds, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "requests_refused_after_deadline": self.requests_refused,
            "requests_abandoned_mid_flight": self.requests_abandoned,
            "cancellation_note": CANCELLATION_NOTE,
        }


#: Closing the HTTP connection does not tell us the server stopped generating,
#: and the runner reports no acknowledgement we could read. So abandonment is
#: recorded as *unconfirmed* rather than asserted -- the honest state, and it
#: means a locally-abandoned request may still have consumed server time.
CANCELLATION_NOTE = (
    "client-side abandonment only; server-side cancellation is NOT confirmed"
)
