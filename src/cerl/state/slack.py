"""Slack-like internal communication subsystem state."""

from __future__ import annotations

from enum import StrEnum

from cerl.core import (
    ApprovalId,
    FrozenMap,
    LogicalInstant,
    MessageId,
    SortedFrozenSet,
    UserId,
)
from cerl.state.common import Frozen, Money


class ApprovalState(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


class SlackUser(Frozen):
    id: UserId
    handle: str
    display_name: str
    roles: SortedFrozenSet[str] = SortedFrozenSet()


class Message(Frozen):
    id: MessageId
    channel: str
    # Stamped by the environment from the acting identity. There is no author
    # field on the post_message action, so an agent cannot forge an approval by
    # writing a message that looks like the approver's (Layer A).
    author: UserId
    text: str
    posted_at: LogicalInstant
    thread_id: MessageId | None = None
    mentions: tuple[UserId, ...] = ()
    # Set when this message is a structured approval request, so the responder
    # engine can match on it without parsing prose.
    approval_subject: str | None = None
    approval_amount: Money | None = None


class Channel(Frozen):
    name: str
    members: SortedFrozenSet[UserId] = SortedFrozenSet()
    # Whether a ticket requester can read this channel. Used by W3's disclosure
    # predicates; present in W2 so the schema does not change between families.
    requester_visible: bool = False


class Approval(Frozen):
    """An approval record.

    ``subject_ref``, ``expires_at``, ``scope_amount_max`` and ``approver`` are
    four *separate* fields because they are the four independent ways an
    approval can be invalid -- the four counterfactual sub-axes. Collapsing them
    into a boolean ``is_valid`` would make the axis unauthorable.
    """

    id: ApprovalId
    request_message_id: MessageId | None
    approver: UserId
    subject_ref: str
    granted_at: LogicalInstant
    expires_at: LogicalInstant | None
    scope_amount_max: Money | None
    state: ApprovalState


class SlackState(Frozen):
    users: FrozenMap[UserId, SlackUser]
    channels: FrozenMap[str, Channel]
    messages: FrozenMap[MessageId, Message] = FrozenMap()
    approvals: FrozenMap[ApprovalId, Approval] = FrozenMap()

    def messages_in(self, channel: str) -> tuple[Message, ...]:
        return tuple(
            m for m in sorted(self.messages.values(), key=lambda m: (m.posted_at, m.id))
            if m.channel == channel
        )

    def approvals_for_subject(self, subject_ref: str) -> tuple[Approval, ...]:
        return tuple(a for a in self.approvals.values() if a.subject_ref == subject_ref)
