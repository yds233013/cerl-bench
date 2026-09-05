"""The append-only action trace and its chain verification."""

from __future__ import annotations

from cerl.core import ChainBroken, Frozen, content_hash
from cerl.diff import Origin, StateDiff
from cerl.trace.entry import GENESIS_HASH, TraceEntry


class ActionTrace(Frozen):
    entries: tuple[TraceEntry, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def head_hash(self) -> str:
        return self.entries[-1].entry_hash if self.entries else GENESIS_HASH

    def append(self, entry: TraceEntry) -> ActionTrace:
        linked = entry.model_copy(
            update={"idx": len(self.entries), "prev_entry_hash": self.head_hash},
        ).sealed()
        return ActionTrace(entries=(*self.entries, linked))

    def agent_entries(self) -> tuple[TraceEntry, ...]:
        """Only agent-origin entries.

        Responder triggers evaluate against exactly this view, which is what
        structurally forbids responder chaining (CLAUDE.md rule 5).
        """
        return tuple(e for e in self.entries if e.origin is Origin.AGENT)

    def responder_entries(self) -> tuple[TraceEntry, ...]:
        return tuple(e for e in self.entries if e.origin is Origin.RESPONDER)

    def business_diffs(self) -> tuple[StateDiff, ...]:
        return tuple(e.business_diff for e in self.entries)

    def verify_chain(self) -> None:
        """Raise ChainBroken if any link or seal fails."""
        previous = GENESIS_HASH
        for position, entry in enumerate(self.entries):
            if entry.idx != position:
                raise ChainBroken(f"entry {position} carries idx {entry.idx}")
            if entry.prev_entry_hash != previous:
                raise ChainBroken(f"entry {position} does not link to its predecessor")
            expected = content_hash(entry.payload_for_hash())
            if entry.entry_hash != expected:
                raise ChainBroken(f"entry {position} seal does not match its contents")
            previous = entry.entry_hash
