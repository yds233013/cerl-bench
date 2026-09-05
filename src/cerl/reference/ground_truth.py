"""Privileged ground truth.

The oracle needs the answer: the resolved branch, the target entities, whether
the approval is really valid. Housing that next to the baselines would mean one
careless import could hand an evaluated agent the answers, with nothing
structural to catch it.

``GroundTruthView`` can only be constructed with a token that ``reference`` alone
can mint, and ``ReferencePolicy.act`` has a *different arity* from
``Agent.act`` -- so the two are not substitutable and mypy enforces that at every
call site (CLAUDE.md rule 6).
"""

from __future__ import annotations

from typing import Any, Protocol

from cerl.actions import Action
from cerl.core import PrivilegeViolation
from cerl.env.observation import Observation
from cerl.scenario.schema import FrozenScenario


class _PrivilegedToken:
    """Unforgeable outside this module: instances are only minted below."""

    __slots__ = ()


_TOKEN = _PrivilegedToken()


def _mint() -> _PrivilegedToken:
    return _TOKEN


class GroundTruthView:
    """The only privileged read path into a scenario's answer."""

    __slots__ = ("_scenario",)

    def __init__(self, scenario: FrozenScenario, token: _PrivilegedToken) -> None:
        if token is not _TOKEN:
            raise PrivilegeViolation(
                "GroundTruthView requires a privileged token minted inside cerl.reference",
            )
        self._scenario = scenario

    @property
    def branch(self) -> str:
        return self._scenario.branch

    @property
    def required_decision(self) -> str:
        return self._scenario.required_decision

    @property
    def variables(self) -> Any:
        return self._scenario.variables

    @property
    def facts(self) -> Any:
        return self._scenario.facts

    @property
    def axes(self) -> Any:
        return self._scenario.axes

    @property
    def scenario(self) -> FrozenScenario:
        return self._scenario

    def var(self, name: str) -> Any:
        return self._scenario.variables[name]


def ground_truth_for(scenario: FrozenScenario) -> GroundTruthView:
    """Mint a privileged view. Callable only from inside ``reference``."""
    return GroundTruthView(scenario, _mint())


class ReferencePolicy(Protocol):
    """Privileged policy interface.

    Deliberately a different signature from ``cerl.agents.base.Agent``: an oracle
    is not substitutable for an agent, nor an agent for an oracle.
    """

    def act(self, observation: Observation, truth: GroundTruthView) -> Action: ...
