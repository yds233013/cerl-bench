"""Freeze-time resolution: axes -> facts -> exactly one branch -> resolved rubric.

After this runs the frozen file is self-contained: the verifier evaluates no
conditionals, because ``applies_when`` and branch scoping have already been
applied here (CLAUDE.md rule 9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cerl.core import FrozenMap, ScenarioDefect, UserId, canonical_json, content_hash
from cerl.scenario.families import registry
from cerl.scenario.generator import GENERATOR_VERSION
from cerl.scenario.schema import FrozenScenario, ScenarioTemplate

FROZEN_DIR = Path("scenarios/frozen")


def templates() -> dict[str, ScenarioTemplate]:
    """All registered templates. Kept as a function so registration order is irrelevant."""
    return registry.all_templates()


def scenario_id(template_id: str, axes: FrozenMap[str, str], seed: int) -> str:
    """A filename that names every axis the family varies.

    Family-specific: see ``registry.register`` for why sharing one family's
    abbreviation map across families silently collides scenarios.
    """
    return f"{template_id}__{registry.slug_for(template_id)(axes)}__s{seed}"


def materialize(
    template_id: str,
    axes: FrozenMap[str, str],
    seed: int,
) -> FrozenScenario:
    """Generate, resolve, and return a self-contained frozen scenario."""
    template = registry.template_for(template_id)
    sid = scenario_id(template_id, axes, seed)
    generated = registry.generator_for(template_id)(seed, axes, sid)

    branch = template.resolve_branch(generated.facts)

    rubric = tuple(item for item in template.rubric if item.resolves_for(branch.name, axes))
    permitted = tuple(
        spec for spec in template.permitted_diffs if spec.resolves_for(branch.name, axes)
    )
    invariants = tuple(
        spec for spec in template.invariants if spec.resolves_for(branch.name, axes)
    )
    if not rubric:
        raise ScenarioDefect(f"{sid}: branch {branch.name} resolved to an empty rubric")

    scenario = FrozenScenario(
        scenario_id=sid,
        template_id=template.id,
        family=template.family,
        schema_version=template.schema_version,
        generator_version=GENERATOR_VERSION,
        root_seed=seed,
        axes=axes,
        facts=generated.facts,
        branch=branch.name,
        required_decision=branch.decision,
        brief=template.brief.format(ticket=generated.variables["ticket"]),
        budget_steps=template.budget_steps,
        agent_user=UserId(str(generated.variables["agent_user"])),
        variables=generated.variables,
        world=generated.world,
        rubric=rubric,
        permitted_diffs=permitted,
        invariants=invariants,
        responders=generated.responders,
    )
    _check_responder_declarations(scenario)
    return scenario


def reachable_responder_rules(scenario: FrozenScenario) -> frozenset[str]:
    """Rules whose guard can actually hold for this instance.

    A guard is a pure function of the request amount, which is fixed per
    scenario, so reachability is decidable at freeze time. Being precise here
    matters: requiring an author to declare a rule that provably cannot fire
    would train them to declare everything, which is how a per-branch
    declaration quietly becomes the global exemption it replaced.
    """
    amount = int(scenario.variables.get("duplicate_amount", 0))
    return frozenset(rule.id for rule in scenario.responders if rule.guard.holds(amount))


def _check_responder_declarations(scenario: FrozenScenario) -> None:
    """Every rule that can fire in this branch must be declared.

    Catching this at freeze time rather than at grading time is what keeps
    "undeclared responder effect" an authoring defect the author fixes, rather
    than a surprise that lands in a verdict.
    """
    declared = {spec.rule for spec in scenario.permitted_diffs if spec.rule}
    missing = reachable_responder_rules(scenario) - declared
    if missing:
        raise ScenarioDefect(
            f"{scenario.scenario_id}: branch {scenario.branch} can fire responder rules "
            f"{sorted(missing)} but declares {sorted(declared)}. Declare them or scope the rule.",
        )


def to_json(scenario: FrozenScenario) -> str:
    payload = scenario.model_dump(mode="json")
    payload["scenario_hash"] = content_hash(payload)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def from_json(text: str) -> FrozenScenario:
    payload = json.loads(text)
    stored = payload.pop("scenario_hash", None)
    scenario = FrozenScenario.model_validate(payload)
    if stored is not None:
        recomputed = content_hash(scenario.model_dump(mode="json"))
        if recomputed != stored:
            raise ScenarioDefect(
                f"{scenario.scenario_id}: content hash mismatch "
                f"(file {stored[:12]}, recomputed {recomputed[:12]})",
            )
    return scenario


def scenario_hash(scenario: FrozenScenario) -> str:
    return content_hash(scenario.model_dump(mode="json"))


def write(scenario: FrozenScenario, directory: Path = FROZEN_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scenario.scenario_id}.json"
    path.write_text(to_json(scenario), encoding="utf-8")
    return path


def load(path: Path) -> FrozenScenario:
    return from_json(path.read_text(encoding="utf-8"))


def canonical_payload(scenario: FrozenScenario) -> str:
    return canonical_json(scenario.model_dump(mode="json"))


def manifest_entry(scenario: FrozenScenario, path: Path) -> dict[str, Any]:
    from cerl.core import hash_text

    return {
        "scenario_id": scenario.scenario_id,
        "file": path.name,
        "sha256": hash_text(path.read_text(encoding="utf-8")),
        "generator_version": scenario.generator_version,
        "template_id": scenario.template_id,
        "family": scenario.family,
        "branch": scenario.branch,
        "axes": scenario.axes.to_dict(),
        "root_seed": scenario.root_seed,
    }
