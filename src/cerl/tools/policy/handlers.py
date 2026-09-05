"""Policy tools (2).

The policy document is retrievable rather than ambient so that "the agent failed
to check the rule" is a real, measurable failure rather than an artifact of us
handing it the rule. Whether a summary also sits in context is an ablation.
"""

from __future__ import annotations

from cerl.actions import PolicyGetRule, PolicySearch, ToolResult, denied, ok
from cerl.state import WorldState
from cerl.tools import interlocks
from cerl.tools.context import ToolContext


def get_rule(
    world: WorldState, action: PolicyGetRule, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    rule = world.policy.rules.get(action.rule_key)
    if rule is None:
        return world, denied(interlocks.NOT_FOUND, f"no policy rule {action.rule_key!r}")
    return world, ok({"rule": {"key": rule.key, "title": rule.title, "text": rule.text}})


def search(
    world: WorldState, action: PolicySearch, _ctx: ToolContext,
) -> tuple[WorldState, ToolResult]:
    needle = action.query.strip().lower()
    hits = [
        {"key": r.key, "title": r.title, "text": r.text}
        for r in sorted(world.policy.rules.values(), key=lambda r: r.key)
        if needle in r.title.lower() or needle in r.text.lower() or needle in r.key.lower()
    ]
    return world, ok({"rules": hits, "count": len(hits)})
