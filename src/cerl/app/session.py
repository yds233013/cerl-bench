"""One workspace session: an environment, and what the operator has observed.

The central discipline here is that **the UI renders only what tool calls have
returned**. A support workspace that displayed the whole world would hand the
operator information the environment never gave them, which would break the same
observation boundary that makes an evaluated agent's score meaningful.

So this holds two things: the live ``CerlEnv``, which nobody may read directly,
and an ``ObservedWorkspace`` accumulated from tool-result payloads. Rendering,
polling and waiting touch only the second, and therefore **cannot advance
logical time**. Time advances when a deliberate action is dispatched, by exactly
the tick cost that action documents.
"""

from __future__ import annotations

from typing import Any

from cerl.actions import Action, ActionKind
from cerl.core import Frozen, FrozenMap
from cerl.env.env import CerlEnv
from cerl.env.observation import Observation
from cerl.scenario.schema import FrozenScenario


class ObservedWorkspace(Frozen):
    """Everything the operator has learned, and nothing else.

    Accumulated from tool results. An empty inbox is the honest starting state:
    you have a ticket reference in the brief and must go and read it.
    """

    tickets: FrozenMap[str, Any] = FrozenMap()
    customers: FrozenMap[str, Any] = FrozenMap()
    charges: FrozenMap[str, Any] = FrozenMap()
    refunds: FrozenMap[str, Any] = FrozenMap()
    messages: tuple[dict[str, Any], ...] = ()
    approvals: FrozenMap[str, Any] = FrozenMap()
    policy_rules: FrozenMap[str, Any] = FrozenMap()
    disputes: FrozenMap[str, Any] = FrozenMap()
    users: FrozenMap[str, Any] = FrozenMap()

    def apply_scalar(
        self, payload: FrozenMap[str, Any], arguments: dict[str, Any],
    ) -> ObservedWorkspace:
        """Fold a scalar result onto the record the action named.

        ``tickets.set_status`` returns ``{"status": ...}`` with no ticket object,
        so a workspace that merged only whole records would keep showing the old
        status after the operator changed it. This attributes the returned value
        to the record the action addressed -- it invents nothing, and uses only
        what the tool actually returned.
        """
        status = payload.get("status")
        ticket_id = arguments.get("ticket_id")
        if not isinstance(status, str) or not isinstance(ticket_id, str):
            return self
        known = self.tickets.get(ticket_id)
        if not isinstance(known, dict):
            return self
        tickets = dict(self.tickets.to_dict())
        tickets[ticket_id] = {**known, "status": status}
        return self.model_copy(update={"tickets": FrozenMap(tickets)})

    def merge(self, payload: FrozenMap[str, Any]) -> ObservedWorkspace:
        """Fold one tool result into the view.

        Additive: a later read never removes something already seen, because an
        operator does not forget a record when a different query omits it.
        """
        data = dict(payload.to_dict())
        tickets = dict(self.tickets.to_dict())
        customers = dict(self.customers.to_dict())
        charges = dict(self.charges.to_dict())
        refunds = dict(self.refunds.to_dict())
        approvals = dict(self.approvals.to_dict())
        rules = dict(self.policy_rules.to_dict())
        disputes = dict(self.disputes.to_dict())
        users = dict(self.users.to_dict())
        messages = list(self.messages)

        def absorb(target: dict[str, Any], value: Any, key: str = "id") -> None:
            if isinstance(value, dict) and value.get(key):
                target[str(value[key])] = value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get(key):
                        target[str(item[key])] = item

        absorb(tickets, data.get("ticket"))
        absorb(tickets, data.get("tickets"))
        absorb(customers, data.get("customer"))
        absorb(customers, data.get("customers"))
        absorb(charges, data.get("charge"))
        absorb(charges, data.get("charges"))
        absorb(refunds, data.get("refund"))
        absorb(refunds, data.get("refunds"))
        absorb(approvals, data.get("approval"))
        absorb(approvals, data.get("approvals"))
        absorb(disputes, data.get("dispute"))
        absorb(disputes, data.get("disputes"))
        absorb(users, data.get("user"))
        absorb(users, data.get("users"))

        rule = data.get("rule")
        if isinstance(rule, dict) and rule.get("key"):
            rules[str(rule["key"])] = rule
        for item in data.get("rules") or []:
            if isinstance(item, dict) and item.get("key"):
                rules[str(item["key"])] = item

        seen = {m.get("id") for m in messages}
        for source in ("message", "messages", "thread"):
            value = data.get(source)
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, dict) and item.get("id") and item["id"] not in seen:
                    messages.append(item)
                    seen.add(item["id"])

        return ObservedWorkspace(
            tickets=FrozenMap(tickets),
            customers=FrozenMap(customers),
            charges=FrozenMap(charges),
            refunds=FrozenMap(refunds),
            messages=tuple(messages),
            approvals=FrozenMap(approvals),
            policy_rules=FrozenMap(rules),
            disputes=FrozenMap(disputes),
            users=FrozenMap(users),
        )


class ActionRecord(Frozen):
    """One dispatched action and what came back. The operator's own history."""

    index: int
    kind: str
    arguments: FrozenMap[str, Any]
    outcome: str
    message: str
    logical_time: int
    #: Set when a backend interlock refused the action. Named so the operator
    #: can see a genuine restriction rather than a silent no-op.
    denied_interlock: str | None = None


class WorkspaceSession:
    """A live session. Not thread-safe; the server serialises per session."""

    def __init__(self, session_id: str, scenario: FrozenScenario, demo_id: str) -> None:
        self.session_id = session_id
        self.demo_id = demo_id
        self.scenario = scenario
        self._env = CerlEnv(scenario)
        self._observation: Observation = self._env.reset()
        self.observed = ObservedWorkspace()
        self.history: list[ActionRecord] = []
        #: Submission tokens already applied, so a double-click, a re-render or
        #: a network retry cannot dispatch the same action twice.
        self._submissions: dict[str, ActionRecord] = {}
        self.notices: list[str] = []

    # -- passive reads: these must never advance logical time ---------------

    @property
    def brief(self) -> str:
        return self.scenario.brief

    @property
    def logical_time(self) -> int:
        return int(self._env.world.clock.now)

    @property
    def step_index(self) -> int:
        return int(self._env.world.meta.step_index)

    @property
    def done(self) -> bool:
        return bool(self._env.done)

    @property
    def declared_outcome(self) -> str | None:
        return self._env.declared_outcome

    @property
    def steps_remaining(self) -> int:
        return max(0, self.scenario.budget_steps - self.step_index)

    # -- the only mutating path -------------------------------------------

    def dispatch(self, action: Action, submission_id: str) -> ActionRecord:
        """Apply one action through the environment's validated dispatcher.

        ``submission_id`` is generated by the client per user gesture. Replaying
        it returns the original record rather than acting again: a retried POST
        is the same submission, and a refund dispatched twice is a real
        duplicate the operator did not intend.
        """
        existing = self._submissions.get(submission_id)
        if existing is not None:
            return existing

        result = self._env.step(action)
        self._observation = result.observation
        payload = result.observation.result.payload if result.observation.result else FrozenMap()
        arguments = {
            k: v for k, v in action.model_dump(mode="json").items() if k != "kind"
        }
        self.observed = self.observed.merge(payload).apply_scalar(payload, arguments)

        tool_result = result.observation.result
        record = ActionRecord(
            index=len(self.history),
            kind=str(action.kind),
            arguments=FrozenMap(
                {k: v for k, v in action.model_dump(mode="json").items() if k != "kind"},
            ),
            outcome=str(tool_result.outcome) if tool_result else "unknown",
            message=tool_result.message if tool_result else "",
            logical_time=self.logical_time,
            denied_interlock=tool_result.denied_interlock if tool_result else None,
        )
        self.history.append(record)
        self._submissions[submission_id] = record
        self.notices = [n.text for n in result.observation.notices]
        return record

    def reset(self) -> None:
        """Back to the exact initial state, including logical time and history."""
        self._env = CerlEnv(self.scenario)
        self._observation = self._env.reset()
        self.observed = ObservedWorkspace()
        self.history = []
        self._submissions = {}
        self.notices = []

    # -- what the operational API is allowed to serve ----------------------

    def view(self) -> dict[str, Any]:
        """The operator-visible session state.

        Deliberately assembled field by field rather than dumped: a dump would
        pick up whatever gets added to the scenario later, and the fields that
        must never appear here -- branch, required decision, rubric, permitted
        diffs, violations, verdict -- are exactly the ones a dump would include.
        """
        return {
            "session_id": self.session_id,
            "demo_id": self.demo_id,
            "scenario_id": self.scenario.scenario_id,
            "brief": self.brief,
            "agent_user": str(self.scenario.agent_user),
            "logical_time": self.logical_time,
            "step_index": self.step_index,
            "steps_remaining": self.steps_remaining,
            "budget_steps": self.scenario.budget_steps,
            "done": self.done,
            "declared_outcome": self.declared_outcome,
            "notices": list(self.notices),
            "observed": {
                "tickets": list(self.observed.tickets.values()),
                "customers": list(self.observed.customers.values()),
                "charges": list(self.observed.charges.values()),
                "refunds": list(self.observed.refunds.values()),
                "approvals": list(self.observed.approvals.values()),
                "messages": list(self.observed.messages),
                "policy_rules": list(self.observed.policy_rules.values()),
                "disputes": list(self.observed.disputes.values()),
                "users": list(self.observed.users.values()),
            },
            "history": [
                {
                    "index": r.index,
                    "kind": r.kind,
                    "arguments": r.arguments.to_dict(),
                    "outcome": r.outcome,
                    "message": r.message,
                    "logical_time": r.logical_time,
                    "denied_interlock": r.denied_interlock,
                }
                for r in self.history
            ],
        }


TERMINAL_KINDS = frozenset(
    {ActionKind.FINISH, ActionKind.ESCALATE, ActionKind.ABSTAIN},
)
