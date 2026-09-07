"""Tool results and the per-entry outcome vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from cerl.core import Frozen, FrozenMap


class Outcome(StrEnum):
    """What happened on a step.

    ``DENIED`` is reserved for Layer-C backend interlocks -- the only layer
    permitted to block. A policy check must never produce it (Invariant B1).
    """

    COMMITTED = "committed"
    READ_ONLY = "read_only"
    DENIED = "denied"
    FAILED = "failed"
    MALFORMED = "malformed"


class ToolResult(Frozen):
    """What the agent observes after acting.

    ``payload`` is the tool's return value as plain JSON. It must never contain
    ground truth (branch, axis values, rubric) -- ``tests/boundaries`` asserts
    this over the rendered observation stream.

    ``FrozenMap`` freezes only its outer level, so nested containers here are
    ordinary JSON and remain mutable. That is deliberate: this object is handed
    to the agent, consumers legitimately annotate what they are given, and the
    payload has to stay plain JSON for canonical serialisation to work -- a
    nested ``FrozenMap`` in a field typed ``Any`` has no pydantic serialiser.

    What must not happen is that editing the returned object changes the
    **sealed trace**. The environment therefore stores its own copy in the trace
    entry (``cerl.env.env._sealed``), so the observation and the record are
    separate objects and mutating one cannot break the other's hash.
    """

    outcome: Outcome
    payload: FrozenMap[str, Any] = FrozenMap()
    message: str = ""
    # Set only when outcome is DENIED; names the frozen Layer-C interlock.
    denied_interlock: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in {Outcome.COMMITTED, Outcome.READ_ONLY}


def ok(payload: Mapping[str, Any] | None = None, message: str = "") -> ToolResult:
    return ToolResult(outcome=Outcome.READ_ONLY, payload=FrozenMap(payload or {}), message=message)


def committed(payload: Mapping[str, Any] | None = None, message: str = "") -> ToolResult:
    return ToolResult(outcome=Outcome.COMMITTED, payload=FrozenMap(payload or {}), message=message)


def denied(interlock: str, message: str) -> ToolResult:
    return ToolResult(outcome=Outcome.DENIED, denied_interlock=interlock, message=message)


def failed(message: str) -> ToolResult:
    return ToolResult(outcome=Outcome.FAILED, message=message)


def malformed(message: str) -> ToolResult:
    return ToolResult(outcome=Outcome.MALFORMED, message=message)
