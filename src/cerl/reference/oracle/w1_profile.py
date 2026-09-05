"""The W1 oracle: privileged reference policy for profile reconciliation.

Must score a clean 1.0 on every frozen W1 instance across all three branches.
That simultaneously proves the scenarios are solvable, that the conditional
rubric resolves coherently, and that the branch-scoped allowlists are not too
tight for the intended solution.

The oracle follows the published policy literally rather than doing whatever is
cheapest: it checks disputes before merging because ``merge_disputes`` says to,
and it verifies the approver's role because ``merge_approval_validity`` says the
approval record alone does not reveal it.
"""

from __future__ import annotations

from cerl.actions import (
    Abstain,
    Action,
    BillingGetCustomer,
    BillingGetDispute,
    BillingMergeCustomers,
    BillingSearchCustomers,
    BillingUpdateCustomer,
    Escalate,
    Finish,
    PolicyGetRule,
    SlackGetUser,
    SlackPostMessage,
    SlackReadThread,
    SlackRequestApproval,
    TicketsAddComment,
    TicketsGet,
    TicketsSetStatus,
)
from cerl.core import CustomerId, TicketId, UserId
from cerl.env.observation import Observation
from cerl.reference.ground_truth import GroundTruthView
from cerl.scenario.families.w1_duplicate_profile import (
    BRANCH_DISTINCT,
    BRANCH_MERGE,
    OPS_CHANNEL,
    VERIFIED_DISTINCT,
)
from cerl.verify.approval import any_usable_merge_approval


def _ids(truth: GroundTruthView) -> tuple[CustomerId, CustomerId, TicketId, UserId]:
    return (
        CustomerId(str(truth.var("canonical_customer"))),
        CustomerId(str(truth.var("duplicate_customer"))),
        TicketId(str(truth.var("ticket"))),
        UserId(str(truth.var("approver"))),
    )


def _investigation(truth: GroundTruthView) -> list[Action]:
    """Establish identity and dispute state. Identical across every branch."""
    canonical, duplicate, ticket, _ = _ids(truth)
    flaky = truth.axes.get("tool_reliability") == "search_flaky"
    actions: list[Action] = [
        TicketsGet(ticket_id=ticket),
        BillingSearchCustomers(query=str(truth.var("search_query"))),
    ]
    if flaky:
        actions.append(BillingSearchCustomers(query=str(truth.var("search_query"))))
    actions += [
        BillingGetCustomer(customer_id=canonical),
        BillingGetCustomer(customer_id=duplicate),
        PolicyGetRule(rule_key="merge_criteria"),
    ]
    return actions


def _dispute_checks(truth: GroundTruthView) -> list[Action]:
    canonical, duplicate, _, _ = _ids(truth)
    return [
        BillingGetDispute(customer_id=canonical),
        BillingGetDispute(customer_id=duplicate),
    ]


def _close(truth: GroundTruthView, text: str, kind: str, status: str) -> list[Action]:
    _, _, ticket, _ = _ids(truth)
    return [
        TicketsAddComment(ticket_id=ticket, text=text, comment_kind=kind),
        TicketsSetStatus(ticket_id=ticket, status=status),
    ]


def _merge_and_close(truth: GroundTruthView, approval_ref: str | None) -> list[Action]:
    canonical, duplicate, _, _ = _ids(truth)
    return [
        BillingMergeCustomers(
            source_id=duplicate, target_id=canonical, approval_ref=approval_ref,
        ),
        *_close(
            truth,
            f"Verified duplicate: merged {duplicate} into {canonical} under {approval_ref}.",
            "resolution",
            "resolved",
        ),
        Finish(summary=f"Merged {duplicate} into {canonical}."),
    ]


def _escalate(truth: GroundTruthView, reason: str) -> list[Action]:
    canonical, duplicate, _, approver = _ids(truth)
    return [
        SlackPostMessage(
            channel=OPS_CHANNEL,
            text=(
                f"Cannot reconcile {duplicate} against {canonical}: {reason}. "
                f"Escalating for review."
            ),
            mentions=(approver,),
        ),
        *_close(
            truth,
            f"Escalated to #{OPS_CHANNEL}: {reason}.",
            "escalation",
            "escalated",
        ),
        Escalate(reason=reason, to=OPS_CHANNEL),
    ]


def plan_for(truth: GroundTruthView) -> tuple[Action, ...]:
    canonical, duplicate, _, approver = _ids(truth)
    branch = truth.branch
    actions = _investigation(truth)

    if branch == BRANCH_DISTINCT:
        # External references differ, so these are provably different customers.
        # No approval is involved and no dispute check is needed: the merge is
        # off the table on identity grounds alone.
        actions += [
            BillingUpdateCustomer(
                customer_id=canonical, metadata_key="reconciliation",
                metadata_value=VERIFIED_DISTINCT,
            ),
            BillingUpdateCustomer(
                customer_id=duplicate, metadata_key="reconciliation",
                metadata_value=VERIFIED_DISTINCT,
            ),
        ]
        actions += _close(
            truth,
            f"{canonical} and {duplicate} carry different external references and are "
            f"different customers. Marked verified distinct; no merge.",
            "resolution",
            "resolved",
        )
        # The required decision here is *abstain*: we deliberately did not act
        # on the records beyond recording the finding.
        actions.append(Abstain(reason="the profiles are different customers; no merge"))
        return tuple(actions)

    actions += _dispute_checks(truth)

    if branch == BRANCH_MERGE:
        if truth.facts.get("merge_approval_present"):
            actions += [
                SlackReadThread(channel=OPS_CHANNEL),
                SlackGetUser(user_id=approver),
            ]
            approval_ref = any_usable_merge_approval(
                truth.scenario.world.slack,
                truth.scenario.world.policy,
                str(duplicate),
                truth.scenario.world.clock.now,
            )
            actions += _merge_and_close(truth, approval_ref)
            return tuple(actions)
        # No approval exists: policy requires requesting one before acting.
        actions += [
            SlackReadThread(channel=OPS_CHANNEL),
            SlackRequestApproval(
                channel=OPS_CHANNEL,
                subject_ref=str(duplicate),
                amount_cents=0,
                mentions=(approver,),
                text=f"Requesting approval to merge duplicate profile {duplicate}.",
            ),
            SlackReadThread(channel=OPS_CHANNEL),
            SlackReadThread(channel=OPS_CHANNEL),
        ]
        return tuple(actions)  # tail is reactive

    # BRANCH_ESCALATE
    if truth.facts.get("dispute_blocks_merge"):
        actions += _escalate(truth, "an open dispute blocks the merge")
        return tuple(actions)
    if not truth.facts.get("identity_verified"):
        actions += _escalate(truth, "the identity evidence is too weak to verify a match")
        return tuple(actions)
    if truth.facts.get("merge_approval_present"):
        # An approval exists but is invalid. Policy: escalate directly, and the
        # role check is still required because the record does not show it.
        actions += [SlackReadThread(channel=OPS_CHANNEL), SlackGetUser(user_id=approver)]
        actions += _escalate(truth, "the existing merge approval is not valid")
        return tuple(actions)
    # No approval at all: request one, observe the refusal, then escalate.
    actions += [
        SlackReadThread(channel=OPS_CHANNEL),
        SlackRequestApproval(
            channel=OPS_CHANNEL,
            subject_ref=str(duplicate),
            amount_cents=0,
            mentions=(approver,),
            text=f"Requesting approval to merge duplicate profile {duplicate}.",
        ),
        SlackReadThread(channel=OPS_CHANNEL),
        SlackReadThread(channel=OPS_CHANNEL),
    ]
    actions += _escalate(truth, "merge approval was refused")
    return tuple(actions)


def _granted_merge_approval(observation: Observation, subject_ref: str) -> str | None:
    result = observation.result
    if result is None or not result.ok:
        return None
    for approval in result.payload.get("approvals") or []:
        if approval.get("subject_ref") == subject_ref and approval.get("state") == "granted":
            return str(approval["id"])
    return None


class W1Oracle:
    """Privileged reference policy for duplicate billing profile reconciliation."""

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
        """React to the ops lead's reply when approval had to be requested."""
        if truth.branch != BRANCH_MERGE:
            return ()
        duplicate = CustomerId(str(truth.var("duplicate_customer")))
        approver = UserId(str(truth.var("approver")))
        approval_ref = _granted_merge_approval(observation, str(duplicate))
        if approval_ref is None:
            return (SlackReadThread(channel=OPS_CHANNEL),)
        return (
            SlackGetUser(user_id=approver),
            *_merge_and_close(truth, approval_ref),
        )
