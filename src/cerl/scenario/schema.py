"""Scenario schema: branches, conditional rubrics, branch-scoped allowlists.

Rubric items and allowlist entries are **branch-conditional** and may further
carry an ``applies_when`` condition on axis values. A flat rubric containing
``refund_exists`` would fail the oracle on every invalid-approval variant, since
the oracle correctly escalates and issues no refund -- which would make "the
oracle scores 1.0" unsatisfiable (CLAUDE.md rule 9).

At **freeze time** the axis assignment resolves to exactly one branch, every
``applies_when`` is evaluated, and the frozen file carries the *already resolved*
rubric, allowlist and required decision. The verifier therefore never evaluates a
conditional.
"""

from __future__ import annotations

from typing import Any

from cerl.core import Frozen, FrozenMap, ScenarioDefect, UserId
from cerl.diff import Origin, PermittedDiff
from cerl.scenario.responders import Decision, ResponderRule
from cerl.state import ConstraintClass, WorldState

ALL_BRANCHES = "all"
SCHEMA_VERSION = 3


def _applies(applies_when: FrozenMap[str, Any], axes: FrozenMap[str, str]) -> bool:
    for axis, expected in applies_when.items():
        actual = axes.get(axis)
        allowed = expected if isinstance(expected, (list, tuple)) else [expected]
        if actual not in [str(a) for a in allowed]:
            return False
    return True


def _in_branch(branches: tuple[str, ...], branch: str) -> bool:
    return ALL_BRANCHES in branches or branch in branches


class BranchSpec(Frozen):
    """One outcome branch. ``when`` matches against freeze-time facts."""

    name: str
    decision: Decision
    when: FrozenMap[str, Any] = FrozenMap()
    note: str = ""

    def matches(self, facts: FrozenMap[str, Any]) -> bool:
        for key, expected in self.when.items():
            actual = facts.get(key)
            if isinstance(expected, (list, tuple)):
                if actual not in list(expected):
                    return False
            elif actual != expected:
                return False
        return True


class RubricItem(Frozen):
    id: str
    branches: tuple[str, ...]
    predicate: str
    args: FrozenMap[str, Any] = FrozenMap()
    applies_when: FrozenMap[str, Any] = FrozenMap()

    def resolves_for(self, branch: str, axes: FrozenMap[str, str]) -> bool:
        return _in_branch(self.branches, branch) and _applies(self.applies_when, axes)


class PermittedDiffSpec(Frozen):
    """An allowlist entry before branch resolution."""

    branches: tuple[str, ...]
    path: str
    origin: Origin = Origin.AGENT
    rule: str | None = None
    constraints: FrozenMap[str, Any] = FrozenMap()
    applies_when: FrozenMap[str, Any] = FrozenMap()
    note: str = ""

    def resolves_for(self, branch: str, axes: FrozenMap[str, str]) -> bool:
        return _in_branch(self.branches, branch) and _applies(self.applies_when, axes)

    def to_permitted(self) -> PermittedDiff:
        return PermittedDiff(
            path=self.path,
            origin=self.origin,
            rule=self.rule,
            constraints=self.constraints,
            note=self.note,
        )


class InvariantSpec(Frozen):
    """A per-step predicate. Latches a COMMITTED violation when it fires.

    Per-step rather than terminal because final-state grading alone is defeated
    by violate-then-revert.

    Branch-scoped for the same reason rubrics are: an operation can be sanctioned
    in one outcome and prohibited in another. W1 merges profiles in the sanctioned
    branch and must never merge in the abstain or escalate branches, so a single
    unscoped "merging is forbidden" invariant would be wrong in both directions.
    """

    predicate: str
    cost_class: ConstraintClass
    branches: tuple[str, ...] = (ALL_BRANCHES,)
    args: FrozenMap[str, Any] = FrozenMap()
    applies_when: FrozenMap[str, Any] = FrozenMap()
    irreversible: bool = False

    def resolves_for(self, branch: str, axes: FrozenMap[str, str]) -> bool:
        return _in_branch(self.branches, branch) and _applies(self.applies_when, axes)


class ScenarioTemplate(Frozen):
    """The authored, unresolved workflow."""

    id: str
    family: str
    schema_version: int = SCHEMA_VERSION
    brief: str
    budget_steps: int = 40
    branches: tuple[BranchSpec, ...]
    rubric: tuple[RubricItem, ...]
    permitted_diffs: tuple[PermittedDiffSpec, ...]
    invariants: tuple[InvariantSpec, ...]

    def resolve_branch(self, facts: FrozenMap[str, Any]) -> BranchSpec:
        """Exactly one branch must match; anything else is a scenario defect."""
        matching = [b for b in self.branches if b.matches(facts)]
        if len(matching) != 1:
            names = [b.name for b in matching]
            raise ScenarioDefect(
                f"{self.id}: expected exactly one matching branch, "
                f"got {names} for facts {dict(facts)}",
            )
        return matching[0]


class FrozenScenario(Frozen):
    """A fully materialised, self-contained instance.

    Committed to ``scenarios/frozen/`` so published numbers are immune to
    generator refactors and scenario review is an ordinary code review over a
    readable diff.
    """

    scenario_id: str
    template_id: str
    family: str
    schema_version: int
    generator_version: str
    #: Which lexicon shard every name in this file came from. Recorded in the
    #: scenario rather than only in the manifest so a file carries its own
    #: provenance: a scenario that ends up in the wrong partition is detectable
    #: from its contents alone.
    lexicon_shard: str = ""
    #: The partition this scenario was generated *for*. Decided before the world
    #: existed, which is what makes shard correctness possible.
    partition: str = ""
    corpus_version: str = ""
    root_seed: int
    axes: FrozenMap[str, str]
    facts: FrozenMap[str, Any]

    branch: str
    required_decision: Decision

    brief: str
    budget_steps: int
    agent_user: UserId
    variables: FrozenMap[str, Any]

    world: WorldState

    # Already resolved for ``branch``; the verifier evaluates no conditionals.
    rubric: tuple[RubricItem, ...]
    permitted_diffs: tuple[PermittedDiffSpec, ...]
    invariants: tuple[InvariantSpec, ...]
    responders: tuple[ResponderRule, ...]

    # Filled by the gold pass, which lives in reference/ and may read ground
    # truth. Used only as the denominator for efficiency reporting.
    oracle_tool_calls: int | None = None

    def permitted(self) -> tuple[PermittedDiff, ...]:
        return tuple(spec.to_permitted() for spec in self.permitted_diffs)

    def rubric_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.rubric)
