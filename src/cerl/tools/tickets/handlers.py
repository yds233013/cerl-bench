"""Ticket tools (5)."""

from __future__ import annotations

from typing import Any

from cerl.actions import (
    TicketsAddComment,
    TicketsAssign,
    TicketsGet,
    TicketsSearch,
    TicketsSetStatus,
    ToolResult,
    committed,
    denied,
    ok,
)
from cerl.core import evolve
from cerl.state import Ticket, TicketComment, WorldState
from cerl.tools import interlocks
from cerl.tools.context import ToolContext


def _ticket_view(ticket: Ticket) -> dict[str, Any]:
    return {
        "id": str(ticket.id),
        "subject": ticket.subject,
        "body": ticket.body,
        "customer_id": str(ticket.customer_id),
        "requester_email": ticket.requester_email,
        "status": ticket.status.value,
        "created_at": int(ticket.created_at),
        "assignee": str(ticket.assignee) if ticket.assignee else None,
        "tags": sorted(ticket.tags),
        "comments": [
            {
                "index": c.index,
                "author": str(c.author),
                "kind": c.kind.value,
                "text": c.text,
                "posted_at": int(c.posted_at),
            }
            for c in ticket.comments
        ],
    }


def get(
    world: WorldState, action: TicketsGet, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    ticket = world.tickets.tickets.get(action.ticket_id)
    if ticket is None:
        return world, denied(interlocks.NOT_FOUND, f"no ticket {action.ticket_id}")
    return world, ok({"ticket": _ticket_view(ticket)})


def search(
    world: WorldState, action: TicketsSearch, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    needle = action.query.strip().lower()
    hits = [
        _ticket_view(t)
        for t in sorted(world.tickets.tickets.values(), key=lambda t: t.id)
        if needle in t.subject.lower() or needle in t.body.lower()
    ]
    return world, ok({"tickets": hits, "count": len(hits)})


def add_comment(
    world: WorldState, action: TicketsAddComment, ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    ticket = world.tickets.tickets.get(action.ticket_id)
    if ticket is None:
        return world, denied(interlocks.NOT_FOUND, f"no ticket {action.ticket_id}")
    # Already validated against the closed set at the action boundary, so there
    # is nothing to coerce and nothing to fall back to. The old fallback to
    # ``note`` silently recorded a different kind than the caller asked for.
    kind = action.comment_kind
    comment = TicketComment(
        index=len(ticket.comments),
        author=ctx.actor,
        kind=kind,
        text=action.text,
        posted_at=world.clock.now,
    )
    updated = evolve(ticket, comments=(*ticket.comments, comment))
    tickets = evolve(world.tickets, tickets=world.tickets.tickets.set(ticket.id, updated))
    return evolve(world, tickets=tickets), committed(
        {"comment_index": comment.index},
    )


def set_status(
    world: WorldState, action: TicketsSetStatus, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    ticket = world.tickets.tickets.get(action.ticket_id)
    if ticket is None:
        return world, denied(interlocks.NOT_FOUND, f"no ticket {action.ticket_id}")
    # Validated at the action boundary. Rejecting it here as a backend "not
    # found" would file a schema violation under Layer-C interlocks, which are
    # for real backend restrictions.
    status = action.status
    updated = evolve(ticket, status=status)
    tickets = evolve(world.tickets, tickets=world.tickets.tickets.set(ticket.id, updated))
    return evolve(world, tickets=tickets), committed({"status": status.value})


def assign(
    world: WorldState, action: TicketsAssign, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    ticket = world.tickets.tickets.get(action.ticket_id)
    if ticket is None:
        return world, denied(interlocks.NOT_FOUND, f"no ticket {action.ticket_id}")
    if action.assignee not in world.slack.users:
        return world, denied(interlocks.NOT_FOUND, f"no user {action.assignee}")
    updated = evolve(ticket, assignee=action.assignee)
    tickets = evolve(world.tickets, tickets=world.tickets.tickets.set(ticket.id, updated))
    return evolve(world, tickets=tickets), committed(
        {"assignee": str(action.assignee)},
    )
