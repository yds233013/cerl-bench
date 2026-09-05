"""Shared state primitives: the frozen base model, money, constraint classes."""

from __future__ import annotations

from cerl.core import CONSTRAINT_ORDER, ConstraintClass, LogicalInstant
from cerl.core.model import Frozen

__all__ = [
    "CONSTRAINT_ORDER",
    "AttemptedViolation",
    "ConstraintClass",
    "Frozen",
    "LatchedViolation",
    "Money",
    "ViolationLog",
    "usd",
]


class Money(Frozen):
    """Integer cents. Never a float -- the verifier compares amounts exactly."""

    cents: int
    currency: str = "USD"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"${self.cents / 100:,.2f}"

    def plus(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(cents=self.cents + other.cents, currency=self.currency)

    def minus(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(cents=self.cents - other.cents, currency=self.currency)

    def _same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")

    def gt(self, other: Money) -> bool:
        self._same_currency(other)
        return self.cents > other.cents

    def lte(self, other: Money) -> bool:
        self._same_currency(other)
        return self.cents <= other.cents


def usd(cents: int) -> Money:
    return Money(cents=cents, currency="USD")


class LatchedViolation(Frozen):
    """A violation the agent actually COMMITTED to state."""

    step_index: int
    logical_time: LogicalInstant
    cost_class: ConstraintClass
    action_kind: str
    irreversible: bool
    detail: str


class AttemptedViolation(Frozen):
    """An unsafe action a Layer-C backend interlock DENIED.

    Kept in a separate sequence from committed violations, and never summed with
    them: one is realised harm to state, the other is evidence of unsafe intent
    that the backend happened to stop.
    """

    step_index: int
    logical_time: LogicalInstant
    cost_class: ConstraintClass
    interlock: str
    action_kind: str
    detail: str


class ViolationLog(Frozen):
    """Append-only, latched. Two independent series (CLAUDE.md rule 2)."""

    committed: tuple[LatchedViolation, ...] = ()
    attempted: tuple[AttemptedViolation, ...] = ()

    def latch_committed(self, violation: LatchedViolation) -> ViolationLog:
        return ViolationLog(
            committed=(*self.committed, violation),
            attempted=self.attempted,
        )

    def latch_attempted(self, violation: AttemptedViolation) -> ViolationLog:
        return ViolationLog(
            committed=self.committed,
            attempted=(*self.attempted, violation),
        )
