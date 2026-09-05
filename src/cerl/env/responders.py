"""The responder engine.

Four properties, each enforced structurally rather than by convention:

* triggers see **only** agent-origin trace entries, so no responder can cause
  another to fire;
* at most **one** responder transition happens per environment step;
* a rule's effect tuple is **atomic** -- one transition regardless of how many
  mutations it contains;
* firing is a pure function of the trace, with no RNG and no wall clock.
"""

from __future__ import annotations

from cerl.actions import ActionKind, ResponderAction
from cerl.core import (
    ApprovalId,
    FrozenMap,
    LogicalInstant,
    MessageId,
    evolve,
)
from cerl.scenario.responders import EffectKind, EffectSpec, ResponderRule, resolve_effect_vars
from cerl.state import Approval, ApprovalState, Message, ScheduledFiring, WorldState, usd
from cerl.tools import mint_runtime_id
from cerl.trace import ActionTrace


def _request_amount(entry_action: object) -> int:
    """The amount a request names, or zero for actions that name none."""
    return int(getattr(entry_action, "amount_cents", 0))


def _trigger_matches(rule: ResponderRule, action: object) -> bool:
    """Whether one agent-origin action satisfies a rule's trigger."""
    if str(getattr(action, "kind", "")) != rule.trigger.tool:
        return False
    if rule.trigger.channel and getattr(action, "channel", None) != rule.trigger.channel:
        return False
    if rule.trigger.mentions:
        mentions = set(getattr(action, "mentions", ()) or ())
        if not set(rule.trigger.mentions) & mentions:
            return False
    return True


def schedule_new_firings(
    world: WorldState,
    rules: tuple[ResponderRule, ...],
    trace: ActionTrace,
) -> WorldState:
    """Schedule any rule whose trigger matches the newest agent-origin entry.

    Only the newest entry is considered, so a rule cannot re-fire on history
    every step. ``agent_entries()`` is the whole no-chaining guarantee: responder
    entries are simply not in the domain this looks at.
    """
    agent_entries = trace.agent_entries()
    if not agent_entries:
        return world
    latest = agent_entries[-1]

    queue = world.responder_queue
    for rule in sorted(rules, key=lambda r: r.id):
        if rule.once and queue.has_pending_or_fired(rule.id):
            continue
        if not _trigger_matches(rule, latest.action):
            continue
        if not rule.guard.holds(_request_amount(latest.action)):
            continue
        queue = queue.scheduled(
            ScheduledFiring(
                rule_id=rule.id,
                fire_at=LogicalInstant(int(latest.logical_time) + rule.delay_ticks),
                triggered_at_step=latest.idx,
            ),
        )
    return evolve(world, responder_queue=queue)


def _apply_effect(world: WorldState, effect: EffectSpec) -> WorldState:
    if effect.kind is EffectKind.POST_MESSAGE:
        message_id = mint_runtime_id(
            MessageId, world, "responder_message", len(world.slack.messages),
        )
        assert effect.author is not None
        message = Message(
            id=message_id,
            channel=effect.channel,
            author=effect.author,
            text=effect.text,
            posted_at=world.clock.now,
        )
        slack = evolve(world.slack, messages=world.slack.messages.set(message_id, message))
        return evolve(world, slack=slack)

    approval_id = mint_runtime_id(
        ApprovalId, world, "responder_approval", len(world.slack.approvals),
    )
    assert effect.approver is not None
    ttl_ticks = (effect.ttl_seconds or 0) // 60
    approval = Approval(
        id=approval_id,
        request_message_id=None,
        approver=effect.approver,
        subject_ref=effect.subject_ref,
        granted_at=world.clock.now,
        expires_at=LogicalInstant(int(world.clock.now) + ttl_ticks) if ttl_ticks else None,
        scope_amount_max=(
            usd(effect.scope_amount_max_cents)
            if effect.scope_amount_max_cents is not None
            else None
        ),
        state=ApprovalState(effect.state),
    )
    slack = evolve(world.slack, approvals=world.slack.approvals.set(approval_id, approval))
    return evolve(world, slack=slack)


def fire_one_due(
    world: WorldState,
    rules: tuple[ResponderRule, ...],
    variables: FrozenMap[str, object],
) -> tuple[WorldState, ResponderAction | None, str | None]:
    """Fire at most one due effect set; leave the rest queued.

    Returning a single optional transition (rather than a list) is the
    one-per-step rule made structural.
    """
    firing = world.responder_queue.due(world.clock.now)
    if firing is None:
        return world, None, None

    rule = next((r for r in rules if r.id == firing.rule_id), None)
    if rule is None:
        return evolve(world, responder_queue=world.responder_queue.consumed(firing)), None, None

    updated = world
    for effect in rule.effect:
        updated = _apply_effect(updated, resolve_effect_vars(effect, variables))
    updated = evolve(updated, responder_queue=updated.responder_queue.consumed(firing))

    action = ResponderAction(
        rule_id=rule.id,
        effect_kinds=tuple(str(e.kind) for e in rule.effect),
        summary=rule.note,
    )
    return updated, action, rule.id


REQUEST_APPROVAL = ActionKind.SLACK_REQUEST_APPROVAL
