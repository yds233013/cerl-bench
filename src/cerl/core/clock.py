"""Logical time.

There is no wall clock anywhere in ``src/cerl`` (CLAUDE.md rule 1). A
``LogicalInstant`` is an integer count of ticks since the scenario epoch, so
elapsed time is a deterministic function of the action sequence and two runs of
the same trajectory agree exactly.

This matters beyond reproducibility: approval validity is evaluated at the
*logical time of the dependent action*, which is what makes "an agent that
dawdles can expire its own approval" a real, testable behaviour rather than an
artifact of when a test happened to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic_core import core_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Self

    from pydantic import GetCoreSchemaHandler

SECONDS_PER_TICK = 60
TICKS_PER_DAY = 24 * 60 * 60 // SECONDS_PER_TICK


class LogicalInstant(int):
    """Ticks since the scenario epoch. One tick is 60 simulated seconds."""

    __slots__ = ()

    def plus(self, ticks: int) -> Self:
        return type(self)(int(self) + ticks)

    def seconds_until(self, other: LogicalInstant) -> int:
        return (int(other) - int(self)) * SECONDS_PER_TICK

    @classmethod
    def from_seconds(cls, seconds: int) -> Self:
        if seconds % SECONDS_PER_TICK:
            raise ValueError(f"{seconds}s is not a whole number of ticks")
        return cls(seconds // SECONDS_PER_TICK)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.int_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                int,
                return_schema=core_schema.int_schema(),
                when_used="always",
            ),
        )


class LogicalClock(BaseModel):
    """An immutable clock. Advancing returns a new clock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now: LogicalInstant

    def advanced(self, ticks: int) -> LogicalClock:
        if ticks < 0:
            raise ValueError("logical time cannot move backwards")
        return LogicalClock(now=self.now.plus(ticks))
