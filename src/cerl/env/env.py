"""The canonical typed environment.

``step`` takes a Pydantic ``Action`` and returns a structured ``StepResult``.
Gymnasium, text and tool-call adapters wrap this; none of them is the source of
truth. Keeping the core typed means the simulator is testable with no model in
the loop, which is what makes offline manifest verification possible.
"""

from __future__ import annotations

from typing import Any

from cerl.actions import (
    Action,
    ActionKind,
    MalformedAction,
    Outcome,
    ToolResult,
    failed,
    malformed,
)
from cerl.actions.models import META_KINDS, TOOL_KINDS
from cerl.core import (
    Frozen,
    FrozenMap,
    LogicalInstant,
    evolve,
    freeze_json,
)
from cerl.diff import Origin, StateDiff
from cerl.env.observation import Notice, Observation, TaskBrief
from cerl.env.responders import fire_one_due, schedule_new_firings
from cerl.env.reward import CostVector, RewardVector, default_scalar
from cerl.scenario.schema import FrozenScenario
from cerl.state import AttemptedViolation, ConstraintClass, FailureKind, WorldState
from cerl.tools import ToolContext, attempted_class_for, handler_for, tick_cost
from cerl.trace import TraceEntry
from cerl.verify.predicates import invariant as invariant_predicates

TERMINAL_KINDS = frozenset(META_KINDS)


class StepResult(Frozen):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: FrozenMap[str, Any]


class CerlEnv:
    """One episode over one frozen scenario."""

    __slots__ = ("_declared_outcome", "_scenario", "_terminated", "_truncated", "_world")

    def __init__(self, scenario: FrozenScenario) -> None:
        self._scenario = scenario
        self._world = scenario.world
        self._terminated = False
        self._truncated = False
        self._declared_outcome: str | None = None

    # -- accessors ---------------------------------------------------------

    @property
    def scenario(self) -> FrozenScenario:
        return self._scenario

    @property
    def world(self) -> WorldState:
        return self._world

    @property
    def declared_outcome(self) -> str | None:
        return self._declared_outcome

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> Observation:
        """Restore the exact initial state.

        Trivially correct because the initial world is an immutable value: this
        is a rebind, not a cleanup protocol.
        """
        self._world = self._scenario.world
        self._terminated = False
        self._truncated = False
        self._declared_outcome = None
        return Observation(
            brief=TaskBrief(
                text=self._scenario.brief,
                ticket_id=str(self._scenario.variables["ticket"]),
                budget_steps=self._scenario.budget_steps,
                agent_user=str(self._scenario.agent_user),
                available_tools=tuple(sorted(TOOL_KINDS)),
            ),
            result=None,
            step_index=0,
            steps_remaining=self._scenario.budget_steps,
            logical_time=self._world.clock.now,
        )

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already finished; call reset()")

        world_before = self._world
        step_index = world_before.meta.step_index
        action_time = world_before.clock.now
        kind = ActionKind(action.kind)

        world_after, result, notices = self._execute(world_before, action, kind)

        # Compute the business diff BEFORE latching, and latch on whether state
        # actually changed rather than on what the result claims. A timed-out
        # call commits its mutation while reporting FAILED; trusting the reported
        # outcome would let an agent violate for free whenever a tool times out.
        agent_diff = self._business_diff(world_before, world_after, Origin.AGENT)

        committed_classes, attempted_classes, world_after = self._latch(
            world_before, world_after, action, result, step_index, action_time,
            mutated=bool(agent_diff),
        )

        world_after = world_after.advanced(tick_cost(kind)).stepped()
        world_after = self._append_entry(
            world_after,
            action=action,
            result=result,
            origin=Origin.AGENT,
            actor=self._scenario.agent_user,
            logical_time=action_time,
            state_hash_before=world_before.state_hash(),
            diff=agent_diff,
            committed_classes=committed_classes,
            attempted_classes=attempted_classes,
        )

        world_after = self._run_responders(world_after)

        self._world = world_after
        self._resolve_termination(kind, world_after)

        return self._build_result(result, notices, committed_classes, attempted_classes)

    # -- internals ---------------------------------------------------------

    def _execute(
        self, world: WorldState, action: Action, kind: ActionKind,
    ) -> tuple[WorldState, ToolResult, tuple[Notice, ...]]:
        if isinstance(action, MalformedAction):
            return world, malformed(action.parse_error or "unparseable action"), (
                Notice(code="malformed", text="action could not be parsed; the step was consumed"),
            )

        if str(kind) in TERMINAL_KINDS:
            return world, ToolResult(outcome=Outcome.READ_ONLY, message=f"{kind} declared"), ()

        tool_id = str(kind)
        call_index = world.tool_health.next_call_index(tool_id)
        failure = world.tool_health.failure_for(tool_id, call_index)
        world = evolve(world, tool_health=world.tool_health.counted(tool_id))

        if failure is FailureKind.UNAVAILABLE:
            return world, failed(f"{tool_id} is temporarily unavailable"), (
                Notice(code="tool_unavailable", text=f"{tool_id} did not respond"),
            )

        handler = handler_for(kind)
        mutated, result = handler(world, action, ToolContext(actor=self._scenario.agent_user))

        if failure is FailureKind.TIMEOUT:
            # The request may have taken effect and the response was lost. The
            # mutation is kept; the agent is told only that it timed out. This
            # is precisely the state that makes list_refunds and idempotency
            # keys matter, so weakening it would delete the retry-safety axis.
            #
            # If the backend would have refused the request, the agent still
            # cannot tell -- but the interlock is preserved on the record, so an
            # unsafe attempt is accounted for rather than hidden by the timeout.
            timed_out = evolve(
                failed(f"{tool_id} timed out; the request may or may not have been applied"),
                denied_interlock=result.denied_interlock,
            )
            return mutated, timed_out, (
                Notice(
                    code="tool_timeout",
                    text=f"{tool_id} timed out - verify current state before retrying",
                ),
            )

        return mutated, result, ()

    def _latch(
        self,
        before: WorldState,
        after: WorldState,
        action: Action,
        result: ToolResult,
        step_index: int,
        action_time: LogicalInstant,
        *,
        mutated: bool,
    ) -> tuple[tuple[ConstraintClass, ...], tuple[ConstraintClass, ...], WorldState]:
        committed = invariant_predicates.evaluate(
            self._scenario.invariants,
            action=action,
            result=result,
            before=before,
            after=after,
            variables=self._scenario.variables,
            step_index=step_index,
            logical_time=action_time,
            mutated=mutated,
        )
        log = after.violations
        for violation in committed:
            log = log.latch_committed(violation)

        attempted: list[ConstraintClass] = []
        # Keyed on the interlock, not the reported outcome: a refusal masked by a
        # timeout is still a refusal, and must still be accounted for.
        if result.denied_interlock:
            cost_class = attempted_class_for(result.denied_interlock)
            if cost_class is not None:
                attempted.append(cost_class)
                log = log.latch_attempted(
                    AttemptedViolation(
                        step_index=step_index,
                        logical_time=action_time,
                        cost_class=cost_class,
                        interlock=result.denied_interlock,
                        action_kind=str(action.kind),
                        detail=result.message,
                    ),
                )
        return (
            tuple(v.cost_class for v in committed),
            tuple(attempted),
            evolve(after, violations=log),
        )

    @staticmethod
    def _business_diff(before: WorldState, after: WorldState, origin: Origin) -> StateDiff:
        # WorldState owns the diff so its memoised document never leaves the
        # object. No copy is taken here, and none is needed: diffing only reads,
        # and DiffOp detaches the values it retains.
        return before.business_diff_to(after, origin=origin)

    @staticmethod
    def _append_entry(
        world: WorldState,
        *,
        action: Any,
        result: ToolResult,
        origin: Origin,
        actor: Any,
        logical_time: LogicalInstant,
        state_hash_before: str,
        diff: StateDiff,
        committed_classes: tuple[ConstraintClass, ...] = (),
        attempted_classes: tuple[ConstraintClass, ...] = (),
        responder_rule: str | None = None,
    ) -> WorldState:
        entry = TraceEntry(
            idx=len(world.trace),
            logical_time=logical_time,
            origin=origin,
            actor=actor,
            action=action,
            # A deep copy, not the same object. The result is also handed to the
            # agent as an observation, and ``FrozenMap`` freezes only its outer
            # level -- so sharing it let an ordinary consumer edit a nested value
            # through the observation and change this sealed entry, breaking its
            # hash. No privileged access was needed, and a consumer merely
            # annotating a returned dictionary would have done it by accident.
            result=_sealed(result),
            outcome=result.outcome,
            violation_classes=committed_classes,
            attempted_classes=attempted_classes,
            denied_interlock=result.denied_interlock,
            responder_rule=responder_rule,
            state_hash_before=state_hash_before,
            state_hash_after=world.state_hash(),
            business_diff=diff,
            prev_entry_hash=world.trace.head_hash,
        )
        return evolve(world, trace=world.trace.append(entry))

    def _run_responders(self, world: WorldState) -> WorldState:
        rules = self._scenario.responders
        if not rules:
            return world
        scheduled = schedule_new_firings(world, rules, world.trace)
        fired, action, rule_id = fire_one_due(scheduled, rules, self._scenario.variables)
        if action is None:
            return fired
        diff = self._business_diff(scheduled, fired, Origin.RESPONDER).with_origin(
            Origin.RESPONDER, rule_id,
        )
        return self._append_entry(
            fired,
            action=action,
            result=ToolResult(outcome=Outcome.COMMITTED, message=f"responder {rule_id} fired"),
            origin=Origin.RESPONDER,
            actor=self._responder_actor(rule_id),
            logical_time=fired.clock.now,
            state_hash_before=scheduled.state_hash(),
            diff=diff,
            responder_rule=rule_id,
        )

    def _responder_actor(self, rule_id: str | None) -> Any:
        rule = next((r for r in self._scenario.responders if r.id == rule_id), None)
        if rule is not None:
            for effect in rule.effect:
                if effect.author is not None:
                    return effect.author
                if effect.approver is not None:
                    return effect.approver
        return self._scenario.agent_user

    def _resolve_termination(self, kind: ActionKind, world: WorldState) -> None:
        if str(kind) in TERMINAL_KINDS:
            self._terminated = True
            self._declared_outcome = {
                ActionKind.FINISH: "act",
                ActionKind.ESCALATE: "escalate",
                ActionKind.ABSTAIN: "abstain",
            }[kind]
            return
        # A catastrophic irreversible violation deliberately does NOT terminate:
        # we need to know whether the agent that deleted the wrong customer also
        # completed the ticket.
        if world.meta.step_index >= self._scenario.budget_steps:
            self._truncated = True

    def _build_result(
        self,
        result: ToolResult,
        notices: tuple[Notice, ...],
        committed_classes: tuple[ConstraintClass, ...],
        attempted_classes: tuple[ConstraintClass, ...],
    ) -> StepResult:
        world = self._world
        committed = CostVector.from_classes(committed_classes)
        attempted = CostVector.from_classes(attempted_classes)
        reward = RewardVector()  # terminal components are computed by the verifier
        observation = Observation(
            brief=None,
            result=result,
            step_index=world.meta.step_index,
            steps_remaining=max(0, self._scenario.budget_steps - world.meta.step_index),
            logical_time=world.clock.now,
            notices=notices,
        )
        return StepResult(
            observation=observation,
            reward=default_scalar(reward, committed),
            terminated=self._terminated,
            truncated=self._truncated,
            info=FrozenMap(
                {
                    # Two separate vectors, never merged (CLAUDE.md rule 2).
                    "committed_costs": committed.as_mapping(),
                    "attempted_costs": attempted.as_mapping(),
                    "state_hash": world.state_hash(),
                    "trace_head": world.trace.head_hash,
                    "declared_outcome": self._declared_outcome,
                },
            ),
        )


def _sealed(result: ToolResult) -> ToolResult:
    """The retained copy of a tool result, isolated from the one the agent gets.

    ``ToolResult.payload`` is a ``FrozenMap[str, Any]``: its outer level is
    immutable, but a nested ``dict`` or ``list`` inside it is not. The same
    object was both returned in the observation and sealed into the trace, so
    editing a nested value through the observation edited the sealed entry and
    broke its hash -- through the ordinary public API, with no privileged access.

    ``freeze_json`` both copies and seals. Copying keeps the observation and the
    record separate; sealing means the record cannot be edited afterwards
    either, which matters because a caller that reaches a trace entry and edits
    its payload in place breaks that entry's hash. The sealed containers are
    ``dict`` and ``list`` subclasses, so the canonical encoding -- and with it
    every historical replay hash -- is byte-identical to before.

    ``evolve`` rather than ``model_copy(update=...)``: the result is revalidated
    as a whole, per the rule that safety-critical packages never install an
    unchecked field.
    """
    return evolve(result, payload=FrozenMap(freeze_json(result.payload.to_dict())))
