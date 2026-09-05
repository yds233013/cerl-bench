"""The complete world state.

One process, one in-memory object graph. No database, no I/O. Reset to an exact
initial state is therefore trivially correct -- re-materialise from the frozen
scenario -- rather than a cleanup protocol we hope works.

``state_hash`` excludes the trace to avoid self-reference; the trace carries its
own chain hash.
"""

from __future__ import annotations

from typing import Any

from pydantic import PrivateAttr

from cerl.core import (
    Frozen,
    LogicalClock,
    content_hash,
    evolve,
)
from cerl.state.billing import BillingState
from cerl.state.common import ViolationLog
from cerl.state.policy import PolicyDocument
from cerl.state.runtime import EpisodeMeta, ResponderQueue, ToolHealthState
from cerl.state.slack import SlackState
from cerl.state.tickets import TicketState
from cerl.trace import ActionTrace


class WorldState(Frozen):
    meta: EpisodeMeta
    clock: LogicalClock
    slack: SlackState
    tickets: TicketState
    billing: BillingState
    policy: PolicyDocument
    tool_health: ToolHealthState = ToolHealthState()
    responder_queue: ResponderQueue = ResponderQueue()
    violations: ViolationLog = ViolationLog()
    trace: ActionTrace = ActionTrace()

    # Memoised derived values. Safe precisely because the model is immutable:
    # a WorldState's document and hash can never change once it exists, and a
    # single step asks for them several times over the same object. Private
    # attributes are excluded from serialization, equality and hashing, so this
    # is invisible to every consumer.
    _document: dict[str, Any] | None = PrivateAttr(default=None)
    _state_hash: str | None = PrivateAttr(default=None)

    def model_copy(self, *, update: Any = None, deep: bool = False) -> WorldState:
        """Copy, discarding memoised derived values.

        Pydantic copies private attributes onto the new instance, so without
        this a mutated world would inherit the *previous* world's document and
        hash -- reporting a stale hash for changed state. That is the failure
        mode memoisation invites, and it is silent, so the reset happens at the
        copy boundary rather than at each call site.
        """
        copied = super().model_copy(update=update, deep=deep)
        copied._document = None  # noqa: SLF001 - resetting our own memo
        copied._state_hash = None  # noqa: SLF001 - resetting our own memo
        return copied

    def as_document(self) -> dict[str, Any]:
        """Canonical JSON view of the whole state (including bookkeeping)."""
        if self._document is None:
            self._document = self.model_dump(mode="json")
        return self._document

    def state_hash(self) -> str:
        """Content hash of the world, excluding the trace.

        The trace is excluded to avoid self-reference; it carries its own chain
        hash.
        """
        if self._state_hash is None:
            document = dict(self.as_document())
            document.pop("trace", None)
            self._state_hash = content_hash(document)
        return self._state_hash

    def advanced(self, ticks: int) -> WorldState:
        return evolve(self, clock=self.clock.advanced(ticks))

    def stepped(self) -> WorldState:
        return evolve(self, meta=self.meta.advanced())
