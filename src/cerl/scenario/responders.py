"""Scripted Slack counterparts.

v1 semantics, all enforced and tested (CLAUDE.md rule 5):

1. **Purely a function of the agent-origin trace.** No RNG, no wall clock.
2. **No chaining.** Triggers are evaluated only against ``origin == "agent"``
   entries, so a responder can never cause another responder to fire.
3. **At most one responder transition per environment step.** If several are due,
   the earliest by ``(fire_at, rule.id)`` fires and the rest stay queued. A
   rule's ``effect`` tuple is atomic -- one transition regardless of how many
   mutations it contains.
4. **No global grading exemption.** A responder diff is permitted only where the
   active branch declares that specific rule.

Responders exist because without them "obtain valid approval through Slack" is
unreachable: the agent could only ever read a pre-existing approval, and the
canonical workflow this benchmark models would be untestable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from cerl.core import Frozen, FrozenMap, UserId


class TriggerKind(StrEnum):
    ACTION_MATCHES = "action_matches"


class GuardKind(StrEnum):
    ALWAYS = "always"
    AMOUNT_AT_MOST = "amount_at_most"
    AMOUNT_GREATER_THAN = "amount_greater_than"


class EffectKind(StrEnum):
    POST_MESSAGE = "post_message"
    CREATE_APPROVAL = "create_approval"


class TriggerSpec(Frozen):
    """Matches one agent-origin action. Structured, never prose-parsing."""

    kind: TriggerKind = TriggerKind.ACTION_MATCHES
    tool: str
    channel: str | None = None
    mentions: tuple[UserId, ...] = ()


class GuardSpec(Frozen):
    """Decides which effect set a trigger produces.

    A failing guard is what makes "requested approval and was refused" a
    reachable, correct-to-escalate state rather than an unreachable one.
    """

    kind: GuardKind = GuardKind.ALWAYS
    limit_cents: int | None = None

    def holds(self, amount_cents: int) -> bool:
        if self.kind is GuardKind.ALWAYS:
            return True
        if self.limit_cents is None:
            return True
        if self.kind is GuardKind.AMOUNT_AT_MOST:
            return amount_cents <= self.limit_cents
        return amount_cents > self.limit_cents


class EffectSpec(Frozen):
    kind: EffectKind
    channel: str = ""
    author: UserId | None = None
    text: str = ""
    approver: UserId | None = None
    subject_ref: str = ""
    state: str = "granted"
    ttl_seconds: int | None = None
    scope_amount_max_cents: int | None = None


class ResponderRule(Frozen):
    id: str
    trigger: TriggerSpec
    guard: GuardSpec = GuardSpec()
    delay_ticks: int = 2
    once: bool = True
    effect: tuple[EffectSpec, ...] = ()
    # Free-form provenance for scenario review; never shown to an agent.
    note: str = ""


class ResponderSet(Frozen):
    rules: tuple[ResponderRule, ...] = ()

    def by_id(self, rule_id: str) -> ResponderRule | None:
        return next((r for r in self.rules if r.id == rule_id), None)


def resolve_effect_vars(effect: EffectSpec, variables: FrozenMap[str, Any]) -> EffectSpec:
    """Substitute ``$.name`` in an effect's subject reference."""
    if effect.subject_ref.startswith("$."):
        name = effect.subject_ref[2:]
        if name in variables:
            return effect.model_copy(update={"subject_ref": str(variables[name])})
    return effect


Decision = Literal["act", "abstain", "escalate"]
