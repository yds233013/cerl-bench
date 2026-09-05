"""The company policy document.

This is *business* state, not bookkeeping: no tool can write it, but closed-world
grading means that if it ever changed the residual would surface it. That is the
correct posture for a field whose mutation would silently rewrite the rules the
agent is graded against (CLAUDE.md rule 4).
"""

from __future__ import annotations

from cerl.core import FrozenMap, SortedFrozenSet, UserId
from cerl.state.common import Frozen, Money


class PolicyRule(Frozen):
    """One retrievable rule, as the agent sees it via ``policy.get_rule``."""

    key: str
    title: str
    text: str


class PolicyDocument(Frozen):
    refund_approval_threshold: Money
    refund_window_days: int
    approval_ttl_seconds: int
    # An approval with less than this much validity left must not be relied on.
    # Declared here, and readable by the agent through ``policy.get_rule``, so
    # that "is this approval usable?" is answerable from visible state alone --
    # never from how long some reference trajectory happens to take.
    minimum_actionable_window_ticks: int
    # role -> users holding it. A "valid" approval requires the approver to hold
    # the role the action needs; that is one of the four independent invalidity
    # modes.
    approver_roles: FrozenMap[str, SortedFrozenSet[UserId]]
    prohibited_ops: SortedFrozenSet[str]
    rules: FrozenMap[str, PolicyRule]

    def holders_of(self, role: str) -> SortedFrozenSet[UserId]:
        return self.approver_roles.get(role, SortedFrozenSet())
