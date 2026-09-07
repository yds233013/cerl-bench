"""Closed vocabularies shared by the action contract and stored state.

``RefundReason`` is a finite set, the way a payment processor's refund reason
is: three values, not free text. It lives in ``core`` for the same reason
``ConstraintClass`` does -- it is domain vocabulary rather than state. ``actions``
sits below ``state`` in the layer stack, so an action model cannot import the
enum from ``state``; defining it twice would let the two drift, which is exactly
the defect this module exists to close.

That drift was real. The action model typed ``reason`` as ``str`` while the
billing handler accepted only these three values, so a schema-valid request
could be rejected by an internal conversion. An agent that sends what the schema
advertises has not misbehaved, and must not be scored as though it had.
"""

from __future__ import annotations

from enum import StrEnum


class TicketStatus(StrEnum):
    """Where a ticket stands. Closed set, mirrored in the agent-visible schema."""

    OPEN = "open"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class CommentKind(StrEnum):
    """What a ticket comment is for. Closed set.

    Typed on the action so an unrecognised kind is rejected at validation. The
    handler used to fall back to ``note``, which silently recorded something
    other than what was asked for.
    """

    RESOLUTION = "resolution"
    ESCALATION = "escalation"
    INFO_REQUEST = "info_request"
    NOTE = "note"


class RefundReason(StrEnum):
    """Why a refund was issued. Closed set, mirrored in the agent-visible schema."""

    DUPLICATE = "duplicate"
    REQUESTED_BY_CUSTOMER = "requested_by_customer"
    FRAUDULENT = "fraudulent"
