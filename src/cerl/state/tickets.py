"""Customer-support ticket subsystem state."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from cerl.core import (
    CommentKind,
    CustomerId,
    FrozenMap,
    LogicalInstant,
    SortedFrozenSet,
    TicketId,
    TicketStatus,
    UserId,
)
from cerl.state.common import Frozen

#: Re-exported: the single definition lives in ``core`` because ``actions``
#: sits below ``state`` and must type these fields too.
__all__ = ["CommentKind", "TicketStatus"]


class TicketComment(Frozen):
    index: int
    author: UserId
    kind: CommentKind
    text: str
    posted_at: LogicalInstant


class Ticket(Frozen):
    id: TicketId
    subject: str
    body: str
    customer_id: CustomerId
    requester_email: str
    status: TicketStatus
    created_at: LogicalInstant
    assignee: UserId | None = None
    tags: SortedFrozenSet[str] = SortedFrozenSet()
    comments: tuple[TicketComment, ...] = ()

    @model_validator(mode="after")
    def _comments_are_an_append_only_log(self) -> Self:
        """Comment indices must be contiguous from zero, in order.

        The comment list is an append-only record; a gap or a repeat would mean
        an entry was dropped or rewritten, which the diff would then misreport.
        """
        expected = list(range(len(self.comments)))
        actual = [c.index for c in self.comments]
        if actual != expected:
            raise ValueError(f"comment indices {actual} are not contiguous from zero")
        return self


class TicketState(Frozen):
    tickets: FrozenMap[TicketId, Ticket]
