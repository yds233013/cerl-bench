"""Bookkeeping state: episode meta, tool health, responder queue.

Everything in this module is excluded from the business projection used for
closed-world side-effect grading (CLAUDE.md rule 4). The clock advances on every
step and tool counters increment on every call; grading them would drown every
episode in residual ops.
"""

from __future__ import annotations

from enum import StrEnum

from cerl.core import FrozenMap, LogicalInstant, SortedFrozenSet
from cerl.state.common import Frozen


class EpisodeMeta(Frozen):
    scenario_id: str
    root_seed: int
    schema_version: int
    step_index: int = 0

    def advanced(self) -> EpisodeMeta:
        return EpisodeMeta(
            scenario_id=self.scenario_id,
            root_seed=self.root_seed,
            schema_version=self.schema_version,
            step_index=self.step_index + 1,
        )


class FailureKind(StrEnum):
    """How an injected tool failure presents."""

    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class ToolHealthState(Frozen):
    """Deterministic failure injection.

    The schedule maps ``tool_id -> {call_index_as_str: FailureKind}`` and is
    materialised into the frozen scenario. Keying on the *per-tool* call index
    rather than a global step counter is load-bearing: it means an unrelated
    interleaved call cannot shift another tool's failures, so two policies face
    the same world and retry semantics have a well-defined meaning
    (CLAUDE.md rule 1).
    """

    schedule: FrozenMap[str, FrozenMap[str, FailureKind]] = FrozenMap()
    call_counts: FrozenMap[str, int] = FrozenMap()

    def next_call_index(self, tool_id: str) -> int:
        return self.call_counts.get(tool_id, 0)

    def failure_for(self, tool_id: str, call_index: int) -> FailureKind | None:
        per_tool = self.schedule.get(tool_id)
        if per_tool is None:
            return None
        return per_tool.get(str(call_index))

    def counted(self, tool_id: str) -> ToolHealthState:
        return ToolHealthState(
            schedule=self.schedule,
            call_counts=self.call_counts.set(tool_id, self.call_counts.get(tool_id, 0) + 1),
        )


class ScheduledFiring(Frozen):
    """A responder rule whose trigger matched and whose effects are pending."""

    rule_id: str
    fire_at: LogicalInstant
    triggered_at_step: int


class ResponderQueue(Frozen):
    """Scheduled-but-unfired responder effects.

    Explicit state so replay reproduces queue contents exactly. Excluded from
    business grading: it is environment scheduling machinery, and grading it
    would penalise an agent for the environment's internals.
    """

    pending: tuple[ScheduledFiring, ...] = ()
    fired: SortedFrozenSet[str] = SortedFrozenSet()

    def scheduled(self, firing: ScheduledFiring) -> ResponderQueue:
        return ResponderQueue(pending=(*self.pending, firing), fired=self.fired)

    def has_pending_or_fired(self, rule_id: str) -> bool:
        return rule_id in self.fired or any(f.rule_id == rule_id for f in self.pending)

    def due(self, now: LogicalInstant) -> ScheduledFiring | None:
        """Return the single next-due firing, or None.

        At most one responder transition happens per environment step, so this
        deliberately returns one item and not a list (CLAUDE.md rule 5). Order is
        ``(fire_at, rule_id)`` so it is total and deterministic.
        """
        candidates = [f for f in self.pending if f.fire_at <= now]
        if not candidates:
            return None
        return min(candidates, key=lambda f: (int(f.fire_at), f.rule_id))

    def consumed(self, firing: ScheduledFiring) -> ResponderQueue:
        remaining = tuple(f for f in self.pending if f != firing)
        return ResponderQueue(
            pending=remaining,
            fired=SortedFrozenSet(self.fired | {firing.rule_id}),
        )
