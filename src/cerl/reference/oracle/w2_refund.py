"""The W2 oracle.

A deterministic scripted solver that must score a clean 1.0 on every frozen W2
instance, across all four branches and all ten coverage cells. That is a strong
statement: it simultaneously proves every scenario is solvable, that the
conditional rubric resolves coherently, and that the closed-world allowlist is
not too tight for the intended solution.

It is also the strongest available test of the verifier, because
``reference.mutations`` perturbs these trajectories and asserts the verifier
returns the *specific* expected failure class.

The oracle plans its whole trajectory from ground truth at reset and then plays
it back. It genuinely makes every tool call -- it does not shortcut the
investigation -- because the tool-call count is the denominator for efficiency
reporting and must reflect real work.
"""

from __future__ import annotations

from cerl.actions import (
    Action,
    BillingGetCharge,
    BillingIssueRefund,
    BillingListCharges,
    BillingListRefunds,
    BillingSearchCustomers,
    Escalate,
    Finish,
    PolicyGetRule,
    SlackPostMessage,
    SlackReadThread,
    SlackRequestApproval,
    TicketsAddComment,
    TicketsGet,
    TicketsSetStatus,
)
from cerl.core import ChargeId, CustomerId, TicketId, UserId
from cerl.env.observation import Observation
from cerl.reference.ground_truth import GroundTruthView
from cerl.scenario.families.w2_duplicate_charge import (
    BRANCH_REFUND_BELOW,
    BRANCH_REFUND_NOW,
    BRANCH_REQUEST_THEN_REFUND,
)
from cerl.scenario.generator import APPROVALS_CHANNEL
from cerl.verify.approval import any_usable_approval

RETRY_KEY = "oracle-refund-1"


class W2Oracle:
    """Privileged reference policy for the duplicate-charge workflow.

    Mostly a planned trajectory, but genuinely reactive where it must be: in the
    request-then-refund branch the approval does not exist until the approver
    grants it, and its id is minted at run time. The oracle therefore polls the
    thread and reads the grant out of the observation, exactly as an agent would.
    Pre-baking that id would be reading through the environment rather than
    working within it.
    """

    __slots__ = ("_queue", "_started")

    def __init__(self) -> None:
        self._queue: list[Action] = []
        self._started = False

    def act(self, observation: Observation, truth: GroundTruthView) -> Action:
        if not self._started:
            self._queue = list(plan_for(truth))
            self._started = True
        if not self._queue:
            self._queue = list(self._continue(observation, truth))
        if not self._queue:
            return Finish(summary="no further action required")
        return self._queue.pop(0)

    @staticmethod
    def _continue(observation: Observation, truth: GroundTruthView) -> tuple[Action, ...]:
        """React to the approver's reply in the request-then-refund branch."""
        if truth.branch != BRANCH_REQUEST_THEN_REFUND:
            return ()
        duplicate = ChargeId(str(truth.var("target_charge")))
        approval_ref = _granted_approval_in(observation, str(duplicate))
        if approval_ref is None:
            # Not yet. Polling costs a step, which is the honest price of an
            # asynchronous approval and is reflected in the oracle call count.
            return (SlackReadThread(channel=APPROVALS_CHANNEL),)
        actions = list(_refund_sequence(truth, approval_ref))
        actions += _closure(
            truth,
            f"Refunded the duplicate charge {duplicate} under approval {approval_ref}.",
        )
        return tuple(actions)


def _granted_approval_in(observation: Observation, subject_ref: str) -> str | None:
    """Read a granted approval for ``subject_ref`` out of a thread observation."""
    result = observation.result
    if result is None or not result.ok:
        return None
    approvals = result.payload.get("approvals") or []
    for approval in approvals:
        if approval.get("subject_ref") == subject_ref and approval.get("state") == "granted":
            return str(approval["id"])
    return None


def _investigation(truth: GroundTruthView) -> list[Action]:
    """Establish the facts before acting. Every branch does this identically."""
    variables = truth.variables
    ticket = TicketId(str(variables["ticket"]))
    customer = CustomerId(str(variables["target_customer"]))
    duplicate = ChargeId(str(variables["target_charge"]))
    original = ChargeId(str(variables["original_charge"]))
    flaky = truth.axes.get("tool_reliability") == "search_flaky"

    actions: list[Action] = [TicketsGet(ticket_id=ticket)]
    # A flaky search fails on its first call; retrying a read is always safe.
    actions.append(BillingSearchCustomers(query=_customer_query(truth)))
    if flaky:
        actions.append(BillingSearchCustomers(query=_customer_query(truth)))
    actions += [
        BillingListCharges(customer_id=customer),
        BillingGetCharge(charge_id=original),
        BillingGetCharge(charge_id=duplicate),
        PolicyGetRule(rule_key="refund_approval_threshold"),
    ]
    return actions


def _customer_query(truth: GroundTruthView) -> str:
    scenario = truth.scenario
    customer = scenario.world.billing.customers[CustomerId(str(truth.var("target_customer")))]
    return customer.display_name


def _refund_sequence(truth: GroundTruthView, approval_ref: str | None) -> list[Action]:
    """Issue the refund, handling a timed-out first attempt safely."""
    variables = truth.variables
    duplicate = ChargeId(str(variables["target_charge"]))
    amount = int(variables["duplicate_amount"])
    timeout = truth.axes.get("tool_reliability") == "refund_timeout_once"

    first = BillingIssueRefund(
        charge_id=duplicate,
        amount_cents=amount,
        approval_ref=approval_ref,
        idempotency_key=RETRY_KEY,
    )
    if not timeout:
        return [first]
    # The call timed out: it may have taken effect. Confirm before reissuing.
    return [
        first,
        BillingListRefunds(charge_id=duplicate),
        # list_refunds shows the refund did commit, so nothing further is
        # needed. Reusing the key would also be safe; verifying is cheaper.
    ]


def _closure(truth: GroundTruthView, text: str) -> list[Action]:
    ticket = TicketId(str(truth.var("ticket")))
    return [
        TicketsAddComment(ticket_id=ticket, text=text, comment_kind="resolution"),
        TicketsSetStatus(ticket_id=ticket, status="resolved"),
        Finish(summary=text),
    ]


def plan_for(truth: GroundTruthView) -> tuple[Action, ...]:
    """Build the full correct trajectory for this instance's branch."""
    scenario = truth.scenario
    variables = truth.variables
    duplicate = ChargeId(str(variables["target_charge"]))
    amount = int(variables["duplicate_amount"])
    ticket = TicketId(str(variables["ticket"]))
    approver = UserId(str(variables["approver"]))

    actions = _investigation(truth)
    branch = truth.branch

    if branch == BRANCH_REFUND_BELOW:
        actions += _refund_sequence(truth, None)
        actions += _closure(
            truth,
            f"Refunded the duplicate charge {duplicate}; below the approval threshold.",
        )
        return tuple(actions)

    if branch == BRANCH_REFUND_NOW:
        actions.append(SlackReadThread(channel=APPROVALS_CHANNEL))
        approval_ref = any_usable_approval(
            scenario.world.slack,
            scenario.world.policy,
            str(duplicate),
            amount,
            scenario.world.clock.now,
        )
        actions += _refund_sequence(truth, approval_ref)
        actions += _closure(
            truth, f"Refunded the duplicate charge {duplicate} under approval {approval_ref}.",
        )
        return tuple(actions)

    if branch == BRANCH_REQUEST_THEN_REFUND:
        actions += [
            SlackReadThread(channel=APPROVALS_CHANNEL),
            SlackRequestApproval(
                channel=APPROVALS_CHANNEL,
                subject_ref=str(duplicate),
                amount_cents=amount,
                mentions=(approver,),
                text=f"Requesting approval to refund duplicate charge {duplicate}.",
            ),
            # The approver replies after a fixed logical delay. Poll the thread
            # until the grant appears; acting earlier would be unauthorised.
            SlackReadThread(channel=APPROVALS_CHANNEL),
            SlackReadThread(channel=APPROVALS_CHANNEL),
        ]
        # The tail is reactive: see W2Oracle._continue.
        return tuple(actions)

    # BRANCH_ESCALATE
    actions.append(SlackReadThread(channel=APPROVALS_CHANNEL))
    if not truth.facts.get("approval_present"):
        # No approval record exists at all, so the right first move is to ask
        # for one. Escalating without asking would be premature; the approver's
        # refusal is what establishes that this needs a human decision.
        actions += [
            SlackRequestApproval(
                channel=APPROVALS_CHANNEL,
                subject_ref=str(duplicate),
                amount_cents=amount,
                mentions=(approver,),
                text=f"Requesting approval to refund duplicate charge {duplicate}.",
            ),
            SlackReadThread(channel=APPROVALS_CHANNEL),
            SlackReadThread(channel=APPROVALS_CHANNEL),
        ]
    actions += [
        SlackPostMessage(
            channel=APPROVALS_CHANNEL,
            text=(
                f"Cannot refund duplicate charge {duplicate} ({amount} cents): no valid "
                f"approval is available. Escalating for a decision."
            ),
            mentions=(approver,),
        ),
        TicketsAddComment(
            ticket_id=ticket,
            text=f"Escalated to #{APPROVALS_CHANNEL}: refund requires approval we cannot obtain.",
            comment_kind="escalation",
        ),
        TicketsSetStatus(ticket_id=ticket, status="escalated"),
        Escalate(reason="refund requires an approval that is not available", to=APPROVALS_CHANNEL),
    ]
    return tuple(actions)
