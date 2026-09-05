"""Slack tools (6).

``post_message`` and ``request_approval`` take **no author field**: the actor is
stamped from the tool context, which the environment owns. That is Layer A -- an
agent cannot forge an approval by writing a message that looks like the
approver's, because the action schema cannot express it.
"""

from __future__ import annotations

from typing import Any

from cerl.actions import (
    SlackGetUser,
    SlackPostMessage,
    SlackReadChannel,
    SlackReadThread,
    SlackRequestApproval,
    SlackSearch,
    ToolResult,
    committed,
    denied,
    ok,
)
from cerl.core import (
    MessageId,
    evolve,
)
from cerl.state import Approval, Message, WorldState, usd
from cerl.tools import interlocks
from cerl.tools.context import ToolContext, mint_runtime_id


def _message_view(message: Message) -> dict[str, Any]:
    return {
        "id": str(message.id),
        "channel": message.channel,
        "author": str(message.author),
        "text": message.text,
        "posted_at": int(message.posted_at),
        "thread_id": str(message.thread_id) if message.thread_id else None,
        "mentions": [str(m) for m in message.mentions],
        "approval_subject": message.approval_subject,
        "approval_amount_cents": (
            message.approval_amount.cents if message.approval_amount else None
        ),
    }


def _approval_view(approval: Approval) -> dict[str, Any]:
    """What the agent sees.

    Deliberately exposes the four independent validity fields rather than a
    computed ``is_valid``: deciding validity is the agent's job, and handing it a
    boolean would collapse the whole approval axis.
    """
    return {
        "id": str(approval.id),
        "approver": str(approval.approver),
        "subject_ref": approval.subject_ref,
        "state": approval.state.value,
        "granted_at": int(approval.granted_at),
        "expires_at": int(approval.expires_at) if approval.expires_at is not None else None,
        "scope_amount_max_cents": (
            approval.scope_amount_max.cents if approval.scope_amount_max else None
        ),
    }


def post_message(
    world: WorldState, action: SlackPostMessage, ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    if action.channel not in world.slack.channels:
        return world, denied(interlocks.NOT_FOUND, f"no channel #{action.channel}")
    message_id = mint_runtime_id(MessageId, world, "message", len(world.slack.messages))
    message = Message(
        id=message_id,
        channel=action.channel,
        author=ctx.actor,  # stamped, never supplied
        text=action.text,
        posted_at=world.clock.now,
        thread_id=action.thread_id,
        mentions=action.mentions,
    )
    slack = evolve(world.slack, messages=world.slack.messages.set(message_id, message))
    return evolve(world, slack=slack), committed(
        {"message": _message_view(message)},
    )


def request_approval(
    world: WorldState, action: SlackRequestApproval, ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    """Post a structured approval request.

    Structured (rather than prose) so the responder engine can match on it
    without parsing natural language, which would be a nondeterminism source.
    """
    if action.channel not in world.slack.channels:
        return world, denied(interlocks.NOT_FOUND, f"no channel #{action.channel}")
    message_id = mint_runtime_id(
        MessageId, world, "approval_request", len(world.slack.messages),
    )
    message = Message(
        id=message_id,
        channel=action.channel,
        author=ctx.actor,
        text=action.text or f"Requesting approval for {action.subject_ref}",
        posted_at=world.clock.now,
        mentions=action.mentions,
        approval_subject=action.subject_ref,
        approval_amount=usd(action.amount_cents),
    )
    slack = evolve(world.slack, messages=world.slack.messages.set(message_id, message))
    return evolve(world, slack=slack), committed(
        {"message": _message_view(message)}, "approval requested",
    )


def read_channel(
    world: WorldState, action: SlackReadChannel, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    if action.channel not in world.slack.channels:
        return world, denied(interlocks.NOT_FOUND, f"no channel #{action.channel}")
    messages = world.slack.messages_in(action.channel)
    return world, ok({"messages": [_message_view(m) for m in messages], "count": len(messages)})


def read_thread(
    world: WorldState, action: SlackReadThread, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    """Read a channel's messages together with any approvals visible there.

    Approvals surface here rather than through a dedicated tool because that is
    how a person would find one: by reading the thread where it was granted.
    """
    if action.channel not in world.slack.channels:
        return world, denied(interlocks.NOT_FOUND, f"no channel #{action.channel}")
    messages = [
        m
        for m in world.slack.messages_in(action.channel)
        if action.thread_id is None or action.thread_id in {m.id, m.thread_id}
    ]
    subjects = {m.approval_subject for m in messages if m.approval_subject}
    approvals = [
        _approval_view(a)
        for a in sorted(world.slack.approvals.values(), key=lambda a: a.id)
        if not subjects or a.subject_ref in subjects
    ]
    return world, ok(
        {
            "messages": [_message_view(m) for m in messages],
            "approvals": approvals,
            "count": len(messages),
        },
    )


def search(
    world: WorldState, action: SlackSearch, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    needle = action.query.strip().lower()
    hits = [
        _message_view(m)
        for m in sorted(world.slack.messages.values(), key=lambda m: (m.posted_at, m.id))
        if needle in m.text.lower() or (m.approval_subject and needle in m.approval_subject.lower())
    ]
    return world, ok({"messages": hits, "count": len(hits)})


def get_user(
    world: WorldState, action: SlackGetUser, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    user = world.slack.users.get(action.user_id)
    if user is None:
        return world, denied(interlocks.NOT_FOUND, f"no user {action.user_id}")
    return world, ok(
        {
            "user": {
                "id": str(user.id),
                "handle": user.handle,
                "display_name": user.display_name,
                "roles": sorted(user.roles),
            },
        },
    )
