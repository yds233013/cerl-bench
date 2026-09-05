"""Family registry: template, generator and freeze plan per workflow.

Until W1 there was one family, so ``freeze`` called the W2 generator directly.
That is no longer tenable and it was never right: a family is a *triple* of
authored template, world generator and instance plan, and the freezing machinery
should not know which one it is holding.

Reference policies are deliberately absent here. ``scenario`` may not import
``reference`` (import-linter contract 4), and the mapping from family to oracle
lives in ``cerl.reference.registry`` instead.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ConfigDict

from cerl.core import Frozen, FrozenMap
from cerl.scenario.generator import GeneratedWorld
from cerl.scenario.schema import ScenarioTemplate

Generator = Callable[[int, FrozenMap[str, str], str], GeneratedWorld]
InstancePlan = Callable[[], tuple[tuple[FrozenMap[str, str], int], ...]]
Slugger = Callable[[FrozenMap[str, str]], str]


class Family(Frozen):
    """One authored workflow and everything needed to materialise it."""

    id: str
    template_id: str
    title: str

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


_TEMPLATES: dict[str, ScenarioTemplate] = {}
_GENERATORS: dict[str, Generator] = {}
_PLANS: dict[str, InstancePlan] = {}
_SLUGS: dict[str, Slugger] = {}
_FAMILY_OF_TEMPLATE: dict[str, str] = {}


def register(
    template: ScenarioTemplate,
    generator: Generator,
    plan: InstancePlan,
    slug: Slugger,
) -> None:
    """Register a family.

    ``slug`` is per-family because a scenario id must name every axis the family
    varies. Sharing one family's abbreviation map with another silently drops the
    axes it does not know about, and distinct scenarios then collide onto one
    filename -- shrinking the corpus without any error.
    """
    _TEMPLATES[template.id] = template
    _GENERATORS[template.id] = generator
    _PLANS[template.id] = plan
    _SLUGS[template.id] = slug
    _FAMILY_OF_TEMPLATE[template.id] = template.family


def template_for(template_id: str) -> ScenarioTemplate:
    return _TEMPLATES[template_id]


def generator_for(template_id: str) -> Generator:
    return _GENERATORS[template_id]


def plan_for(template_id: str) -> InstancePlan:
    return _PLANS[template_id]


def slug_for(template_id: str) -> Slugger:
    return _SLUGS[template_id]


def template_ids() -> tuple[str, ...]:
    return tuple(sorted(_TEMPLATES))


def families() -> tuple[str, ...]:
    return tuple(sorted(set(_FAMILY_OF_TEMPLATE.values())))


def template_ids_for_family(family: str) -> tuple[str, ...]:
    return tuple(
        sorted(tid for tid, fam in _FAMILY_OF_TEMPLATE.items() if fam == family)
    )


def all_templates() -> dict[str, ScenarioTemplate]:
    return dict(_TEMPLATES)
