"""Customer-support ticket subsystem state."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import model_validator

from cerl.core import (
    CustomerId,
    FrozenMap,
    LogicalInstant,
    SortedFrozenSet,
    TicketId,
    UserId,
)
from cerl.state.common import Frozen


class TicketStatus(StrEnum):
    OPEN = "open"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class CommentKind(StrEnum):
    RESOLUTION = "resolution"
    ESCALATION = "escalation"
    INFO_REQUEST = "info_request"
    NOTE = "note"


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
