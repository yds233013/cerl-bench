"""Tool dispatch and logical costs.

Every tool costs exactly one logical tick; meta-actions cost zero. A uniform
cost keeps elapsed time a simple function of the call count, which is what makes
"the approval expires N steps after it is granted" an axis an author can reason
about without simulating the trajectory first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cerl.actions import ActionKind, ToolResult
from cerl.state import WorldState
from cerl.tools.billing import handlers as billing
from cerl.tools.context import ToolContext
from cerl.tools.policy import handlers as policy
from cerl.tools.slack import handlers as slack
from cerl.tools.tickets import handlers as tickets

ToolHandler = Callable[[WorldState, Any, ToolContext], tuple[WorldState, ToolResult]]

TOOL_TICK_COST = 1
META_TICK_COST = 0

REGISTRY: dict[ActionKind, ToolHandler] = {
    ActionKind.SLACK_POST_MESSAGE: slack.post_message,
    ActionKind.SLACK_READ_CHANNEL: slack.read_channel,
    ActionKind.SLACK_READ_THREAD: slack.read_thread,
    ActionKind.SLACK_SEARCH: slack.search,
    ActionKind.SLACK_REQUEST_APPROVAL: slack.request_approval,
    ActionKind.SLACK_GET_USER: slack.get_user,
    ActionKind.TICKETS_GET: tickets.get,
    ActionKind.TICKETS_SEARCH: tickets.search,
    ActionKind.TICKETS_ADD_COMMENT: tickets.add_comment,
    ActionKind.TICKETS_SET_STATUS: tickets.set_status,
    ActionKind.TICKETS_ASSIGN: tickets.assign,
    ActionKind.BILLING_GET_CUSTOMER: billing.get_customer,
    ActionKind.BILLING_SEARCH_CUSTOMERS: billing.search_customers,
    ActionKind.BILLING_LIST_CHARGES: billing.list_charges,
    ActionKind.BILLING_GET_CHARGE: billing.get_charge,
    ActionKind.BILLING_LIST_REFUNDS: billing.list_refunds,
    ActionKind.BILLING_ISSUE_REFUND: billing.issue_refund,
    ActionKind.BILLING_UPDATE_CUSTOMER: billing.update_customer,
    ActionKind.BILLING_GET_DISPUTE: billing.get_dispute,
    ActionKind.BILLING_MERGE_CUSTOMERS: billing.merge_customers,
    ActionKind.BILLING_DELETE_CUSTOMER: billing.delete_customer,
    ActionKind.POLICY_GET_RULE: policy.get_rule,
    ActionKind.POLICY_SEARCH: policy.search,
}

TOOL_COUNT = len(REGISTRY)


def handler_for(kind: ActionKind) -> ToolHandler:
    return REGISTRY[kind]


def tick_cost(kind: ActionKind) -> int:
    return META_TICK_COST if kind not in REGISTRY else TOOL_TICK_COST
