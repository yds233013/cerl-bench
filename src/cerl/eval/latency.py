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
