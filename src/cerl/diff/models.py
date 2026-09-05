"""Diff operations and their canonical ordering."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from cerl.core import Frozen


class DiffOpKind(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


class Origin(StrEnum):
    """Who caused a change.

    Load-bearing for grading: responder-generated changes are matched against
    *declared* allowlist entries rather than waved through globally, and an
    undeclared one is a scenario authoring defect rather than agent misbehaviour
    (CLAUDE.md rule 4).
    """

    AGENT = "agent"
    RESPONDER = "responder"
    SYSTEM = "system"


class DiffOp(Frozen):
    op: DiffOpKind
    path: str
    before: Any = None
    after: Any = None
    origin: Origin = Origin.AGENT
    responder_rule: str | None = None

    def sort_key(self) -> tuple[str, str]:
        return (self.path, self.op.value)


class StateDiff(Frozen):
    """A canonically ordered sequence of operations."""

    ops: tuple[DiffOp, ...] = ()

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self) -> Any:
        return iter(self.ops)

    def __bool__(self) -> bool:
        return bool(self.ops)

    @classmethod
    def of(cls, ops: tuple[DiffOp, ...]) -> StateDiff:
        return cls(ops=tuple(sorted(ops, key=lambda o: o.sort_key())))

    def with_origin(self, origin: Origin, responder_rule: str | None = None) -> StateDiff:
        return StateDiff.of(
            tuple(
                op.model_copy(update={"origin": origin, "responder_rule": responder_rule})
                for op in self.ops
            ),
        )

    def paths(self) -> tuple[str, ...]:
        return tuple(op.path for op in self.ops)
