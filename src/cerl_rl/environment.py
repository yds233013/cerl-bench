"""Episode semantics and reward. **No torch.**

Separated from ``rollout`` so the parts that define what an episode *is* --
where it stops, what it scores, how model text becomes an action -- can be
exercised, audited and replayed without a deep-learning stack present. The
model-driven rollout imports from here, so there is exactly one definition of
the control flow rather than one for training and a second for review.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import TypeAdapter, ValidationError

from cerl.actions import Action, MalformedAction
from cerl.agents.tool_schemas import all_tool_schemas, tool_name_to_kind
from cerl.env.env import CerlEnv, StepResult
from cerl.env.reward import CostVector, RewardVector, default_scalar
from cerl.scenario.schema import FrozenScenario

_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

#: The three actions that end an episode.
TERMINAL_KINDS = frozenset({"finish", "escalate", "abstain"})


class ActionCategory(StrEnum):
    """What an action was, for counting purposes.

    v1 counted every non-malformed action as a ``tool_call``, terminal
    declarations included, which disagrees with the benchmark verifier: it
    reported 15 tool calls for a validation phase the verifier scores as 10, and
    191 against 150 in training. Three categories, counted separately, so a
    number can be compared with the verifier's instead of quietly redefining it.
    """

    TOOL_CALL = "tool_call"
    TERMINAL = "terminal"
    MALFORMED = "malformed"


def categorise(action: Action) -> ActionCategory:
    kind = str(action.kind)
    if kind == "malformed":
        return ActionCategory.MALFORMED
    if kind in TERMINAL_KINDS:
        return ActionCategory.TERMINAL
    return ActionCategory.TOOL_CALL


def tool_menu() -> str:
    lines = []
    for schema in all_tool_schemas():
        args = ", ".join(sorted(schema["input_schema"]["properties"]))
        lines.append(f"- {schema['name']}({args})")
    return "\n".join(lines)


def episode_reward(verdict: Any) -> float:
    """The pilot's training objective: ``cerl``'s own ``default_scalar``.

    The formula is untouched. What this does is assemble its two arguments for a
    *finished episode*, because the per-step reward the environment returns is
    deliberately empty -- terminal components come from the verifier, so a
    mid-episode scalar would always be zero.

    ``r_efficiency`` is left at its default because ``default_scalar`` does not
    weight it. Committed costs enter through ``CostVector``. Attempted costs
    cannot, because ``default_scalar`` has no parameter for them -- the Phase-1
    decision (CLAUDE.md rule 2), inherited here rather than quietly revisited.
    """
    committed = CostVector.from_classes(tuple(v.cost_class for v in verdict.violations))
    reward_vector = RewardVector(
        r_outcome=1.0 if verdict.correct_final_state else 0.0,
        r_task=float(verdict.task_completion),
        r_decision=1.0 if verdict.decision_correct else 0.0,
    )
    return float(default_scalar(reward_vector, committed))


#: Every tool name the public schema advertises. An allowlist, because
#: ``tool_name_to_kind`` is a string substitution and will happily transform a
#: name that no tool has.
def _public_tool_kinds() -> dict[str, str]:
    return {schema["name"]: tool_name_to_kind(schema["name"]) for schema in all_tool_schemas()}


PUBLIC_TOOLS: dict[str, str] = _public_tool_kinds()

#: Field names the caller may never supply. ``kind`` is the discriminator the
#: action union dispatches on: allowing it as an argument let a request name one
#: tool and execute another.
RESERVED_ARGUMENTS = frozenset({"kind"})


def parse_action(text: str) -> Action:
    """Model text -> a validated Action, or a scored MalformedAction.

    Exactly the benchmark's contract: an unparseable or invalid turn is a real,
    recorded action that consumes a step. It is never retried, repaired or
    silently dropped -- any of those would make the malformed-turn rate
    unmeasurable, and that rate is what the earlier 4B run actually measured.

    Three hardenings over the first version, each closing a way for the parsed
    action to be something other than what the text asked for:

    * The tool is looked up in the **public allowlist**, not transformed by
      string substitution, so a name no tool has cannot become a plausible kind.
    * ``kind`` is **reserved**. ``{"kind": kind, **arguments}`` previously let an
      argument named ``kind`` overwrite the discriminator, so
      ``{"tool": "tickets__get", "arguments": {"kind": "billing.delete_customer",
      ...}}`` parsed as a *customer deletion*. The core's interlocks and policy
      grading were never bypassed -- but the advertised contract was, and an
      action was recorded that the text did not request.
    * ``arguments`` must be an **object**. ``payload.get("arguments") or {}``
      turned ``0``, ``""``, ``[]`` and ``null`` into an empty argument set,
      silently inventing a call the model did not make.
    """
    snippet = text.strip()
    start, end = snippet.find("{"), snippet.rfind("}")
    if start == -1 or end <= start:
        return MalformedAction(
            raw=snippet[:400],
            parse_error="the turn contained no JSON object; one action per turn is required",
        )
    try:
        payload = json.loads(snippet[start : end + 1])
    except json.JSONDecodeError as error:
        return MalformedAction(raw=snippet[:400], parse_error=f"invalid JSON: {error.msg}")
    if not isinstance(payload, dict) or "tool" not in payload:
        return MalformedAction(raw=snippet[:400], parse_error="no 'tool' key in the object")

    name = str(payload["tool"])
    kind = PUBLIC_TOOLS.get(name)
    if kind is None:
        return MalformedAction(
            raw=snippet[:400],
            parse_error=f"unknown tool {name!r}; it is not in the advertised tool list",
        )

    if "arguments" not in payload:
        arguments: Any = {}
    else:
        arguments = payload["arguments"]
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            return MalformedAction(
                raw=snippet[:400],
                parse_error=(
                    f"'arguments' must be an object, got {type(arguments).__name__}"
                ),
            )

    reserved = RESERVED_ARGUMENTS & set(arguments)
    if reserved:
        return MalformedAction(
            raw=snippet[:400],
            parse_error=(
                f"{sorted(reserved)} may not be supplied as arguments; the tool is "
                f"chosen by 'tool' alone"
            ),
        )

    try:
        return _ACTION_ADAPTER.validate_python({"kind": kind, **arguments})
    except ValidationError as error:
        return MalformedAction(
            raw=snippet[:400],
            parse_error=f"{kind}: {error.errors()[0].get('msg', 'invalid arguments')}",
        )


def drive(
    scenario: FrozenScenario,
    choose: Callable[[int, Any], Action | None],
    *,
    max_actions: int,
    after_step: Callable[[int, Action, Any, CerlEnv], None] | None = None,
) -> tuple[CerlEnv, list[Action], str | None]:
    """Run one episode. **The single definition of the pilot's episode semantics.**

    A fresh :class:`CerlEnv` per call, built from the frozen scenario, so every
    episode starts from the exact initial state and no rollout can inherit an
    earlier one's damage.

    Two stopping rules, and both matter for reading any result from this pilot:

    * a **terminal action** (``finish`` / ``escalate`` / ``abstain``) ends the
      episode, and its kind is recorded as the declared outcome;
    * otherwise the episode ends after ``max_actions`` environment steps and is
      recorded as **step-limited** -- a real, kept outcome, not an error.

    ``choose`` returns the next action, or ``None`` to stop early (used when a
    scripted source runs out, or when the model's context is exhausted).

    ``after_step`` is called once per executed action, immediately after the
    environment applies it and while its outcome is known. That timing is what
    lets a caller durably record a turn *as it completes*, rather than at the
    end of an episode -- an episode that is interrupted half way through has
    still really executed its earlier turns, and losing them loses evidence the
    environment actually produced.
    """
    env = CerlEnv(scenario)
    env.reset()
    actions: list[Action] = []
    declared: str | None = None
    observation: Any = env.reset()
    for step in range(max_actions):
        action = choose(step, observation)
        if action is None:
            break
        result: StepResult = env.step(action)
        actions.append(action)
        if after_step is not None:
            # The env is passed so the hook can read the trace entry the step
            # just wrote, without the caller having to smuggle a reference out.
            after_step(step, action, result, env)
        observation = result.observation
        if result.terminated:
            declared = str(action.kind)
            break
    return env, actions, declared


def scripted(actions: tuple[Action, ...]) -> Callable[[int, Any], Action | None]:
    """An action source that replays a fixed list -- e.g. a gold trajectory."""

    def choose(step: int, _observation: Any) -> Action | None:
        return actions[step] if step < len(actions) else None

    return choose


def terminal_index(actions: tuple[Action, ...]) -> int | None:
    """Position of the first terminal action, or ``None`` if there is none."""
    for index, action in enumerate(actions):
        if str(action.kind) in TERMINAL_KINDS:
            return index
    return None
