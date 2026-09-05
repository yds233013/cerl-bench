"""Hash-chained trace entries.

Each entry commits to its predecessor, so a post-hoc edit anywhere in the record
is detectable. For an open benchmark accepting submissions, a trace whose chain
verifies and whose terminal ``state_hash_after`` matches the submitted score is
strong evidence the run actually happened.

Committed and attempted violation classes are **separate tuples on every
entry**, so the distinction survives at the rawest level of the record and any
downstream analysis can recover it without re-deriving from the verdict
(CLAUDE.md rule 2).
"""

from __future__ import annotations

from typing import Any

from cerl.actions import Action, ResponderAction, ToolResult
from cerl.actions.results import Outcome
from cerl.core import ConstraintClass, Frozen, LogicalInstant, UserId, content_hash
from cerl.diff import Origin, StateDiff

GENESIS_HASH = "0" * 64


class TraceEntry(Frozen):
    idx: int
    logical_time: LogicalInstant
    origin: Origin
    # Stamped by the environment from the acting identity, never agent-supplied.
    actor: UserId
    action: Action | ResponderAction
    result: ToolResult
    outcome: Outcome
    violation_classes: tuple[ConstraintClass, ...] = ()
    attempted_classes: tuple[ConstraintClass, ...] = ()
    denied_interlock: str | None = None
    responder_rule: str | None = None
    state_hash_before: str
    state_hash_after: str
    business_diff: StateDiff
    prev_entry_hash: str
    entry_hash: str = ""

    def payload_for_hash(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("entry_hash", None)
        return data

    def sealed(self) -> TraceEntry:
        """Return this entry with its chain hash computed."""
        return self.model_copy(update={"entry_hash": content_hash(self.payload_for_hash())})

    @property
    def action_kind(self) -> str:
        return str(self.action.kind)
