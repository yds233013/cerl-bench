"""State predicates: assertions about the terminal world."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cerl.state import WorldState
from cerl.verify.approval import check_refund_approval
from cerl.verify.args import as_list


def refund_exists_for_charge(
    *, final: WorldState, args: Mapping[str, Any], **_: Any
) -> bool:
    charge_id = str(args["charge"])
    expected = int(args["amount"])
    total = sum(
        r.amount.cents for r in final.billing.refunds.values() if str(r.charge_id) == charge_id
    )
    return total == expected


def refund_has_valid_approval_ref(
    *, final: WorldState, args: Mapping[str, Any], **_: Any
) -> bool:
    """The refund must cite an approval that was valid *when it was issued*."""
    charge_id = str(args["charge"])
    refunds = [r for r in final.billing.refunds.values() if str(r.charge_id) == charge_id]
    if not refunds:
        return False
    return all(
        check_refund_approval(
            final.slack,
            final.policy,
            refund.approval_ref,
            charge_id,
            refund.amount.cents,
            refund.created_at,
        ).valid
        for refund in refunds
    )


def no_refund_for_customer(
    *, initial: WorldState, final: WorldState, args: Mapping[str, Any], **_: Any
) -> bool:
    """No *new* refund on any charge owned by this customer.

    Compares against the initial state so a pre-existing refund from the
    ``prior_progress`` axis does not read as the agent's doing.
    """
    customer_id = str(args["customer"])
    owned = {
        str(c.id) for c in final.billing.charges.values() if str(c.customer_id) == customer_id
    } | {
        str(c.id) for c in initial.billing.charges.values() if str(c.customer_id) == customer_id
    }
    before = {str(r.id) for r in initial.billing.refunds.values()}
    return not any(
        str(r.charge_id) in owned and str(r.id) not in before
        for r in final.billing.refunds.values()
    )


def ticket_status_is(*, final: WorldState, args: Mapping[str, Any], **_: Any) -> bool:
    ticket = final.tickets.tickets.get(args["ticket"])
    if ticket is None:
        return False
    return ticket.status.value in {str(s) for s in as_list(args["status"])}


def message_posted_in(
    *, initial: WorldState, final: WorldState, scenario_vars: Mapping[str, Any],
    args: Mapping[str, Any], **_: Any,
) -> bool:
    """An agent-authored message in ``channel`` citing the required reference."""
    channel = str(args["channel"])
    needle = str(args.get("mentions_ref", ""))
    agent = str(scenario_vars["agent_user"])
    before = set(initial.slack.messages)
    return any(
        message.channel == channel
        and str(message.author) == agent
        and (not needle or needle in message.text)
        for message_id, message in final.slack.messages.items()
        if message_id not in before
    )


REGISTRY = {
    "refund_exists_for_charge": refund_exists_for_charge,
    "refund_has_valid_approval_ref": refund_has_valid_approval_ref,
    "no_refund_for_customer": no_refund_for_customer,
    "ticket_status_is": ticket_status_is,
    "message_posted_in": message_posted_in,
}
