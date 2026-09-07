"""Diff operations and their canonical ordering."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import field_validator

from cerl.core import Frozen, freeze_json


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
    """One change, with the values on either side of it.

    ``before`` and ``after`` are typed ``Any`` because a document node can be a
    scalar, an object or an array. Pydantic therefore does not rebuild them, so
    a freshly built op holds the **caller's own objects** -- and an op built
    from a state document would hold live references into that document. Ops
    are retained in the trace, so mutating one would reach back into state that
    is supposed to be frozen.

    The values are therefore sealed on construction, which both detaches them
    from the caller's objects and makes the retained value itself immutable.
    Detaching alone would not be enough: ops are hashed into the trace entry, so
    a caller editing a *recorded* op in place breaks that entry's seal. This is
    per-op over a small subtree, not per-document, so it is affordable on the
    per-step path in a way that copying the whole world is not.
    """

    op: DiffOpKind
    path: str
    before: Any = None
    after: Any = None
    origin: Origin = Origin.AGENT
    responder_rule: str | None = None

    @field_validator("before", "after", mode="after")
    @classmethod
    def _detach_and_seal(cls, value: Any) -> Any:
        return freeze_json(value)

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
