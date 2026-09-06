"""The action space: 23 tools + 3 meta-actions + 1 error = 27 kinds.

A data-only package importing nothing but ``core``. Actions are *values*; the
environment is their interpreter. Keeping them here (rather than inside ``env``)
lets ``trace`` hold a fully typed ``Action`` without a package cycle, so the most
safety-critical record in the project stays typed rather than degrading to an
untyped payload dict.

Layer A of the enforcement model lives here: an unsafe thing the schema cannot
express is unrepresentable. Most visibly, ``SlackPostMessage`` has **no author
field** -- the environment stamps the acting identity -- so an agent cannot forge
an approval by writing a message that looks like the approver's.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from cerl.core import (
    ChargeId,
    CustomerId,
    DisputeId,
    Frozen,
    MessageId,
    RefundReason,
    TicketId,
    UserId,
)


class ActionKind(StrEnum):
    """Every action kind. The registry test asserts this enum has 27 members."""

    SLACK_POST_MESSAGE = "slack.post_message"
    SLACK_READ_CHANNEL = "slack.read_channel"
    SLACK_READ_THREAD = "slack.read_thread"
    SLACK_SEARCH = "slack.search"
    SLACK_REQUEST_APPROVAL = "slack.request_approval"
    SLACK_GET_USER = "slack.get_user"

    TICKETS_GET = "tickets.get"
    TICKETS_SEARCH = "tickets.search"
    TICKETS_ADD_COMMENT = "tickets.add_comment"
    TICKETS_SET_STATUS = "tickets.set_status"
    TICKETS_ASSIGN = "tickets.assign"

    BILLING_GET_CUSTOMER = "billing.get_customer"
    BILLING_SEARCH_CUSTOMERS = "billing.search_customers"
    BILLING_LIST_CHARGES = "billing.list_charges"
    BILLING_GET_CHARGE = "billing.get_charge"
    BILLING_LIST_REFUNDS = "billing.list_refunds"
    BILLING_ISSUE_REFUND = "billing.issue_refund"
    BILLING_UPDATE_CUSTOMER = "billing.update_customer"
    BILLING_GET_DISPUTE = "billing.get_dispute"
    BILLING_MERGE_CUSTOMERS = "billing.merge_customers"
    BILLING_DELETE_CUSTOMER = "billing.delete_customer"

    POLICY_GET_RULE = "policy.get_rule"
    POLICY_SEARCH = "policy.search"

    FINISH = "finish"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"

    MALFORMED = "malformed"


TOOL_KINDS: frozenset[str] = frozenset(
    k.value
    for k in ActionKind
    if k not in {ActionKind.FINISH, ActionKind.ESCALATE, ActionKind.ABSTAIN, ActionKind.MALFORMED}
)

META_KINDS: frozenset[str] = frozenset(
    {ActionKind.FINISH.value, ActionKind.ESCALATE.value, ActionKind.ABSTAIN.value},
)

MUTATING_KINDS: frozenset[str] = frozenset(
    {
        ActionKind.SLACK_POST_MESSAGE.value,
        ActionKind.SLACK_REQUEST_APPROVAL.value,
        ActionKind.TICKETS_ADD_COMMENT.value,
        ActionKind.TICKETS_SET_STATUS.value,
        ActionKind.TICKETS_ASSIGN.value,
        ActionKind.BILLING_ISSUE_REFUND.value,
        ActionKind.BILLING_UPDATE_CUSTOMER.value,
        ActionKind.BILLING_MERGE_CUSTOMERS.value,
        ActionKind.BILLING_DELETE_CUSTOMER.value,
    },
)


class BaseAction(Frozen):
    """Common base. ``kind`` is the discriminator."""


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


class SlackPostMessage(BaseAction):
    # No author field, by design (Layer A).
    kind: Literal[ActionKind.SLACK_POST_MESSAGE] = ActionKind.SLACK_POST_MESSAGE
    channel: str
    text: str
    thread_id: MessageId | None = None
    mentions: tuple[UserId, ...] = ()


class SlackReadChannel(BaseAction):
    kind: Literal[ActionKind.SLACK_READ_CHANNEL] = ActionKind.SLACK_READ_CHANNEL
    channel: str


class SlackReadThread(BaseAction):
    kind: Literal[ActionKind.SLACK_READ_THREAD] = ActionKind.SLACK_READ_THREAD
    channel: str
    thread_id: MessageId | None = None


class SlackSearch(BaseAction):
    kind: Literal[ActionKind.SLACK_SEARCH] = ActionKind.SLACK_SEARCH
    query: str


class SlackRequestApproval(BaseAction):
    kind: Literal[ActionKind.SLACK_REQUEST_APPROVAL] = ActionKind.SLACK_REQUEST_APPROVAL
    channel: str
    subject_ref: str
    amount_cents: int
    mentions: tuple[UserId, ...] = ()
    text: str = ""


class SlackGetUser(BaseAction):
    kind: Literal[ActionKind.SLACK_GET_USER] = ActionKind.SLACK_GET_USER
    user_id: UserId


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


class TicketsGet(BaseAction):
    kind: Literal[ActionKind.TICKETS_GET] = ActionKind.TICKETS_GET
    ticket_id: TicketId


class TicketsSearch(BaseAction):
    kind: Literal[ActionKind.TICKETS_SEARCH] = ActionKind.TICKETS_SEARCH
    query: str


class TicketsAddComment(BaseAction):
    kind: Literal[ActionKind.TICKETS_ADD_COMMENT] = ActionKind.TICKETS_ADD_COMMENT
    ticket_id: TicketId
    text: str
    comment_kind: str = "note"


class TicketsSetStatus(BaseAction):
    kind: Literal[ActionKind.TICKETS_SET_STATUS] = ActionKind.TICKETS_SET_STATUS
    ticket_id: TicketId
    status: str


class TicketsAssign(BaseAction):
    kind: Literal[ActionKind.TICKETS_ASSIGN] = ActionKind.TICKETS_ASSIGN
    ticket_id: TicketId
    assignee: UserId


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------


class BillingGetCustomer(BaseAction):
    kind: Literal[ActionKind.BILLING_GET_CUSTOMER] = ActionKind.BILLING_GET_CUSTOMER
    customer_id: CustomerId


class BillingSearchCustomers(BaseAction):
    kind: Literal[ActionKind.BILLING_SEARCH_CUSTOMERS] = ActionKind.BILLING_SEARCH_CUSTOMERS
    query: str


class BillingListCharges(BaseAction):
    kind: Literal[ActionKind.BILLING_LIST_CHARGES] = ActionKind.BILLING_LIST_CHARGES
    customer_id: CustomerId


class BillingGetCharge(BaseAction):
    kind: Literal[ActionKind.BILLING_GET_CHARGE] = ActionKind.BILLING_GET_CHARGE
    charge_id: ChargeId


class BillingListRefunds(BaseAction):
    """The only way to learn whether a timed-out refund actually committed.

    Load-bearing for the retry-safety cells, not decoration.
    """

    kind: Literal[ActionKind.BILLING_LIST_REFUNDS] = ActionKind.BILLING_LIST_REFUNDS
    charge_id: ChargeId


class BillingIssueRefund(BaseAction):
    kind: Literal[ActionKind.BILLING_ISSUE_REFUND] = ActionKind.BILLING_ISSUE_REFUND
    charge_id: ChargeId
    amount_cents: int
    # Typed as the enum so an out-of-vocabulary value is rejected where the
    # schema promises it will be -- at validation -- and becomes a scored
    # MalformedAction rather than surfacing from inside a tool handler.
    reason: RefundReason = RefundReason.DUPLICATE
    approval_ref: str | None = None
    idempotency_key: str | None = None


class BillingUpdateCustomer(BaseAction):
    kind: Literal[ActionKind.BILLING_UPDATE_CUSTOMER] = ActionKind.BILLING_UPDATE_CUSTOMER
    customer_id: CustomerId
    metadata_key: str
    metadata_value: str


class BillingGetDispute(BaseAction):
    kind: Literal[ActionKind.BILLING_GET_DISPUTE] = ActionKind.BILLING_GET_DISPUTE
    dispute_id: DisputeId | None = None
    customer_id: CustomerId | None = None


class BillingMergeCustomers(BaseAction):
    """Irreversible. Exists so a prohibited side effect is actually causable."""

    kind: Literal[ActionKind.BILLING_MERGE_CUSTOMERS] = ActionKind.BILLING_MERGE_CUSTOMERS
    source_id: CustomerId
    target_id: CustomerId
    approval_ref: str | None = None


class BillingDeleteCustomer(BaseAction):
    """Irreversible and prohibited in every W2 branch."""

    kind: Literal[ActionKind.BILLING_DELETE_CUSTOMER] = ActionKind.BILLING_DELETE_CUSTOMER
    customer_id: CustomerId


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PolicyGetRule(BaseAction):
    kind: Literal[ActionKind.POLICY_GET_RULE] = ActionKind.POLICY_GET_RULE
    rule_key: str


class PolicySearch(BaseAction):
    kind: Literal[ActionKind.POLICY_SEARCH] = ActionKind.POLICY_SEARCH
    query: str


# ---------------------------------------------------------------------------
# Meta + error
# ---------------------------------------------------------------------------


class Finish(BaseAction):
    kind: Literal[ActionKind.FINISH] = ActionKind.FINISH
    summary: str = ""


class Escalate(BaseAction):
    kind: Literal[ActionKind.ESCALATE] = ActionKind.ESCALATE
    reason: str = ""
    to: str = ""


class Abstain(BaseAction):
    kind: Literal[ActionKind.ABSTAIN] = ActionKind.ABSTAIN
    reason: str = ""


class MalformedAction(BaseAction):
    """Unparseable output is a first-class action that consumes a step.

    It is not an exception: raising would let a harness silently retry, hiding a
    real and interesting behaviour and breaking determinism.
    """

    kind: Literal[ActionKind.MALFORMED] = ActionKind.MALFORMED
    raw: str = ""
    parse_error: str = ""


Action: TypeAlias = Annotated[
    SlackPostMessage
    | SlackReadChannel
    | SlackReadThread
    | SlackSearch
    | SlackRequestApproval
    | SlackGetUser
    | TicketsGet
    | TicketsSearch
    | TicketsAddComment
    | TicketsSetStatus
    | TicketsAssign
    | BillingGetCustomer
    | BillingSearchCustomers
    | BillingListCharges
    | BillingGetCharge
    | BillingListRefunds
    | BillingIssueRefund
    | BillingUpdateCustomer
    | BillingGetDispute
    | BillingMergeCustomers
    | BillingDeleteCustomer
    | PolicyGetRule
    | PolicySearch
    | Finish
    | Escalate
    | Abstain
    | MalformedAction,
    Field(discriminator="kind"),
]


ACTION_TYPES: tuple[type[BaseAction], ...] = (
    SlackPostMessage,
    SlackReadChannel,
    SlackReadThread,
    SlackSearch,
    SlackRequestApproval,
    SlackGetUser,
    TicketsGet,
    TicketsSearch,
    TicketsAddComment,
    TicketsSetStatus,
    TicketsAssign,
    BillingGetCustomer,
    BillingSearchCustomers,
    BillingListCharges,
    BillingGetCharge,
    BillingListRefunds,
    BillingIssueRefund,
    BillingUpdateCustomer,
    BillingGetDispute,
    BillingMergeCustomers,
    BillingDeleteCustomer,
    PolicyGetRule,
    PolicySearch,
    Finish,
    Escalate,
    Abstain,
    MalformedAction,
)


class ResponderAction(Frozen):
    """What a scripted responder did. Never produced by an agent."""

    kind: Literal["responder"] = "responder"
    rule_id: str
    effect_kinds: tuple[str, ...]
    summary: str = ""
