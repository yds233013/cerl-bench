"""The unprivileged agent interface.

``act`` takes an ``Observation`` and nothing else. There is no second parameter
through which ground truth could arrive, which is the point: the restriction is
structural, not a convention that a future refactor could quietly relax.

``agents`` may not import ``reference``, ``verify`` or ``scenario``
(import-linter contract 1).
"""

from __future__ import annotations

import inspect
from typing import Protocol, runtime_checkable

from cerl.actions import Action
from cerl.env.observation import Observation


def is_unprivileged_agent(candidate: object) -> bool:
    """Whether ``candidate`` can only ever receive an observation.

    ``isinstance`` against a ``runtime_checkable`` Protocol checks method
    *presence*, not signature, so it would happily accept a privileged oracle
    that also happens to define ``act``. The arity is the thing that actually
    separates them, so that is what this checks. Static enforcement is mypy's
    job; this is the runtime backstop for the evaluation harness.
    """
    act = getattr(candidate, "act", None)
    if act is None or not callable(act):
        return False
    parameters = [
        name
        for name, param in inspect.signature(act).parameters.items()
        if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    return parameters == ["observation"]


@runtime_checkable
class Agent(Protocol):
    """An evaluated policy. Sees observations; never state, truth or rubric."""

    def act(self, observation: Observation) -> Action: ...


class ScriptedAgent:
    """Replays a fixed action list. Used for adversarial and golden fixtures."""

    __slots__ = ("_actions", "_fallback", "_index")

    def __init__(self, actions: tuple[Action, ...], fallback: Action | None = None) -> None:
        self._actions = actions
        self._index = 0
        self._fallback = fallback

    def act(self, observation: Observation) -> Action:  # noqa: ARG002
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
            return action
        if self._fallback is None:
            raise IndexError("scripted agent exhausted and no fallback was provided")
        return self._fallback
