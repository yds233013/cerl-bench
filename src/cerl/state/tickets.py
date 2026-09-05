"""Customer-support ticket subsystem state."""

from __future__ import annotations

from enum import StrEnum

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


class TicketState(Frozen):
    tickets: FrozenMap[TicketId, Ticket]
