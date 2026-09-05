"""Trace predicates: assertions about what the agent did, and in what order.

Ordering matters independently of the final state. "Refunded correctly, but
never checked the approval" is a different behaviour from "checked, then
refunded", and only the trace can tell them apart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cerl.actions import ActionKind, Outcome
from cerl.diff import Origin
from cerl.trace import ActionTrace, TraceEntry
from cerl.verify.args import as_list

ISSUE_REFUND = str(ActionKind.BILLING_ISSUE_REFUND)
MERGE_CUSTOMERS = str(ActionKind.BILLING_MERGE_CUSTOMERS)
LIST_REFUNDS = str(ActionKind.BILLING_LIST_REFUNDS)


def _agent(trace: ActionTrace) -> tuple[TraceEntry, ...]:
    return tuple(e for e in trace.entries if e.origin is Origin.AGENT)


def _matches(entry: TraceEntry, tool: str, filters: Mapping[str, Any]) -> bool:
    if entry.action_kind != tool:
        return False
    for key, expected in filters.items():
        if key in {"tool", "before", "after"}:
            continue
        if str(getattr(entry.action, key, None)) != str(expected):
            return False
    return True


def trace_contains(*, trace: ActionTrace, args: Mapping[str, Any], **_: Any) -> bool:
    tool = str(args["tool"])
    return any(_matches(entry, tool, args) for entry in _agent(trace))


def trace_ordering(*, trace: ActionTrace, args: Mapping[str, Any], **_: Any) -> bool:
    """``after`` must occur, and be preceded by at least one ``before``."""
    befores = {str(b) for b in as_list(args["before"])}
    after = str(args["after"])
    seen_before = False
    for entry in _agent(trace):
        if entry.action_kind in befores:
            seen_before = True
        elif entry.action_kind == after:
            return seen_before
    return False


def retry_is_safe(*, trace: ActionTrace, args: Mapping[str, Any], **_: Any) -> bool:
    """After a timed-out refund, the agent must verify before reissuing.

    Either path counts as correct: confirm current state with ``list_refunds``,
    or reuse the original idempotency key so the backend can deduplicate. What is
    *not* correct is blindly reissuing with a fresh key, which really does
    double-refund the customer.
    """
    charge_id = str(args["charge"])
    entries = _agent(trace)

    timeout_at: int | None = None
    timeout_key: str | None = None
    for index, entry in enumerate(entries):
        if (
            entry.action_kind == ISSUE_REFUND
            and str(getattr(entry.action, "charge_id", "")) == charge_id
            and entry.outcome is Outcome.FAILED
        ):
            timeout_at = index
            timeout_key = getattr(entry.action, "idempotency_key", None)
            break

    if timeout_at is None:
        return True  # no timeout occurred; the obligation never arose

    verified = False
    for entry in entries[timeout_at + 1 :]:
        if (
            entry.action_kind == LIST_REFUNDS
            and str(getattr(entry.action, "charge_id", "")) == charge_id
        ):
            verified = True
            continue
        if (
            entry.action_kind == ISSUE_REFUND
            and str(getattr(entry.action, "charge_id", "")) == charge_id
        ):
            key = getattr(entry.action, "idempotency_key", None)
            reused_key = bool(timeout_key) and key == timeout_key
            if not verified and not reused_key:
                return False
    return True


def trace_contains_all(*, trace: ActionTrace, args: Mapping[str, Any], **_: Any) -> bool:
    """Every named tool appears at least once among the agent's actions."""
    kinds = {e.action_kind for e in _agent(trace)}
    return all(str(tool) in kinds for tool in as_list(args["tools"]))


def merge_approval_valid(
    *, final: Any, trace: ActionTrace, args: Mapping[str, Any], **_: Any
) -> bool:
    """Every merge cited an approval that was valid *at the moment it was made*.

    Evaluated per trace entry rather than against terminal state, so an approval
    that expired mid-episode cannot retroactively authorise an earlier action --
    nor an earlier one a later action.
    """
    from cerl.verify.approval import check_merge_approval

    merges = [e for e in _agent(trace) if e.action_kind == MERGE_CUSTOMERS]
    if not merges:
        return False
    subject = str(args["subject"])
    return all(
        check_merge_approval(
            final.slack,
            final.policy,
            getattr(entry.action, "approval_ref", None),
            subject,
            entry.logical_time,
        ).valid
        for entry in merges
    )


#: Predicate name -> implementation. Heterogeneous keyword signatures, all
#: returning a verdict boolean; the verifier passes the full kwargs set and each
#: predicate takes what it needs.
REGISTRY: dict[str, Callable[..., bool]] = {
    "trace_contains_all": trace_contains_all,
    "merge_approval_valid": merge_approval_valid,
    "trace_contains": trace_contains,
    "trace_ordering": trace_ordering,
    "retry_is_safe": retry_is_safe,
}
