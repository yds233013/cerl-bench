"""Deterministic control baselines.

Reference points, not competitors. Each is unprivileged -- ``act`` takes an
``Observation`` and nothing else -- and each is *deterministic*, so its numbers
are a fixed property of the corpus rather than of a run.

They exist to make three different claims checkable:

* **always-escalate** is the degenerate-safety reference. Any arm whose safety
  score is indistinguishable from this one has bought safety with inaction.
* **always-abstain** and **always-finish** are its siblings: a policy that
  always declares one outcome will succeed on exactly the scenarios requiring
  that outcome, and nowhere else.
* **random-valid-action** is the lower bound. It establishes that the rubric is
  not passable by chance.

None of them reads privileged state; the ticket id comes from the public brief.
"""

from __future__ import annotations

from collections.abc import Callable

from cerl.actions import (
    Abstain,
    Action,
    BillingGetCustomer,
    BillingListCharges,
    BillingSearchCustomers,
    Escalate,
    Finish,
    PolicyGetRule,
    PolicySearch,
    SlackPostMessage,
    TicketsAddComment,
    TicketsGet,
    TicketsSearch,
    TicketsSetStatus,
)
from cerl.core import CustomerId, TicketId, derive_below
from cerl.env.observation import Observation

#: Fallback when the published escalation rule names no channel.
DEFAULT_ESCALATION_CHANNEL = "billing-approvals"


class _SingleOutcomeAgent:
    """Reads the ticket, then always declares the same outcome."""

    __slots__ = ("_declared", "_ticket")

    def __init__(self) -> None:
        self._ticket: str | None = None
        self._declared = False

    def _terminal(self) -> Action:
        raise NotImplementedError

    def act(self, observation: Observation) -> Action:
        if observation.brief is not None:
            self._ticket = observation.brief.ticket_id
        if not self._declared and self._ticket is not None:
            self._declared = True
            return TicketsGet(ticket_id=TicketId(self._ticket))
        return self._terminal()


class AlwaysEscalateAgent(_SingleOutcomeAgent):
    """The degenerate-safety reference."""

    name = "always_escalate"

    def _terminal(self) -> Action:
        return Escalate(reason="escalating without investigating", to="billing-approvals")


class AlwaysAbstainAgent(_SingleOutcomeAgent):
    """Never acts, never escalates."""

    name = "always_abstain"

    def _terminal(self) -> Action:
        return Abstain(reason="abstaining without investigating")


class AlwaysFinishAgent(_SingleOutcomeAgent):
    """Declares the task done without doing anything."""

    name = "always_finish"

    def _terminal(self) -> Action:
        return Finish(summary="declared complete without investigating")


class RandomValidAgent:
    """Seeded pseudo-random choice over schema-valid actions.

    The lower bound. Draws are counter-based key derivations, so the sequence is
    a function of the seed and the step index alone -- reproducible, and
    identical across processes and Python versions.
    """

    name = "random_valid"

    __slots__ = ("_seed", "_step", "_ticket")

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._step = 0
        self._ticket: str | None = None

    def act(self, observation: Observation) -> Action:
        if observation.brief is not None:
            self._ticket = observation.brief.ticket_id
        ticket = TicketId(self._ticket) if self._ticket else None

        pool: list[Action] = [
            TicketsSearch(query="refund"),
            BillingSearchCustomers(query="acme"),
            PolicySearch(query="approval"),
            Finish(summary="done"),
            Escalate(reason="unsure", to="billing-approvals"),
            Abstain(reason="unsure"),
        ]
        if ticket is not None:
            pool.append(TicketsGet(ticket_id=ticket))

        index = derive_below(self._seed, "random_valid_agent", self._step, len(pool))
        self._step += 1
        return pool[index]




class InvestigateThenEscalateAgent:
    """Escalate everything, but do the escalation work properly.

    The meaningful degenerate-safety ceiling. A policy that merely *declares*
    escalation scores zero even on escalation branches, because escalating
    correctly requires posting the reference and setting the ticket status --
    escalation is not free. This control does that work, so its score is the
    real upper bound for "when in doubt, escalate", and any arm that does not
    beat it has bought its safety with inaction.

    Entirely unprivileged: the ticket comes from the public brief, the channel
    is read out of the published escalation policy, and the identifiers are
    whatever the investigation surfaced. It cites everything it found rather
    than reasoning about which reference matters -- a degenerate policy, by
    construction.
    """

    name = "investigate_then_escalate"

    __slots__ = ("_channel", "_customer", "_ids", "_plan", "_ticket")

    def __init__(self) -> None:
        self._ticket: str | None = None
        self._customer: str | None = None
        self._channel: str | None = None
        self._ids: list[str] = []
        self._plan = 0

    @staticmethod
    def _channel_from(text: str) -> str | None:
        for token in text.replace("\n", " ").split():
            if token.startswith("#") and len(token) > 1:
                return token[1:].strip(".,;:")
        return None

    def _harvest(self, observation: Observation) -> None:
        """Collect every identifier and channel name the observation exposed."""
        result = observation.result
        if result is None or not result.ok:
            return
        payload = result.payload.to_dict()

        ticket = payload.get("ticket")
        if isinstance(ticket, dict):
            self._customer = str(ticket.get("customer_id") or "") or self._customer

        for key in ("charges", "customers", "refunds"):
            for item in payload.get(key) or []:
                if isinstance(item, dict) and item.get("id"):
                    self._ids.append(str(item["id"]))
        for key in ("charge", "customer"):
            item = payload.get(key)
            if isinstance(item, dict) and item.get("id"):
                self._ids.append(str(item["id"]))

        rule = payload.get("rule")
        if isinstance(rule, dict):
            self._channel = self._channel_from(str(rule.get("text", ""))) or self._channel

    def act(self, observation: Observation) -> Action:
        if observation.brief is not None:
            self._ticket = observation.brief.ticket_id
        self._harvest(observation)
        ticket = TicketId(self._ticket) if self._ticket else None
        step = self._plan
        self._plan += 1

        if ticket is None:
            return Abstain(reason="no ticket in the brief")
        return self._step(step, ticket)

    # A dispatch table rather than a chain of numeric comparisons: the plan is
    # data, so the sequence is readable in one place and reordering it is a
    # one-line edit rather than a renumbering.
    def _step(self, step: int, ticket: TicketId) -> Action:
        """Dispatch one plan step.

        A table of bound methods rather than a chain of numeric comparisons: the
        investigation sequence reads in one place, and reordering it does not
        mean renumbering a ladder of ``if step == n``.
        """
        plan: tuple[Callable[[TicketId], Action], ...] = (
            self._read_ticket,
            self._read_escalation_rule,
            self._list_charges,
            self._read_customer,
            self._post_escalation,
            self._comment,
            self._set_status,
        )
        if step >= len(plan):
            return Escalate(
                reason="escalating every ticket",
                to=self._channel or DEFAULT_ESCALATION_CHANNEL,
            )
        return plan[step](ticket)

    def _read_ticket(self, ticket: TicketId) -> Action:
        return TicketsGet(ticket_id=ticket)

    def _read_escalation_rule(self, _ticket: TicketId) -> Action:
        return PolicyGetRule(rule_key="escalation")

    def _list_charges(self, ticket: TicketId) -> Action:
        if not self._customer:
            return self._read_escalation_rule(ticket)
        return BillingListCharges(customer_id=CustomerId(self._customer))

    def _read_customer(self, ticket: TicketId) -> Action:
        if not self._customer:
            return self._read_escalation_rule(ticket)
        return BillingGetCustomer(customer_id=CustomerId(self._customer))

    def _post_escalation(self, _ticket: TicketId) -> Action:
        refs = " ".join(sorted(set(self._ids))) or "no identifiers found"
        return SlackPostMessage(
            channel=self._channel or DEFAULT_ESCALATION_CHANNEL,
            text=f"Escalating for review. References seen: {refs}",
        )

    def _comment(self, ticket: TicketId) -> Action:
        return TicketsAddComment(
            ticket_id=ticket,
            text="Escalated for human review.",
            comment_kind="escalation",
        )

    def _set_status(self, ticket: TicketId) -> Action:
        return TicketsSetStatus(ticket_id=ticket, status="escalated")


CONTROL_AGENTS: dict[str, type] = {
    AlwaysEscalateAgent.name: AlwaysEscalateAgent,
    AlwaysAbstainAgent.name: AlwaysAbstainAgent,
    AlwaysFinishAgent.name: AlwaysFinishAgent,
    InvestigateThenEscalateAgent.name: InvestigateThenEscalateAgent,
}
