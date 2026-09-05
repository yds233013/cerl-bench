"""Scenario authoring: templates, axes, mutators, freezing.

Inert data: ``scenario`` imports no runtime package (import-linter contract 4).
"""

from cerl.scenario import axes
from cerl.scenario.freeze import (
    TEMPLATES,
    from_json,
    load,
    manifest_entry,
    materialize,
    reachable_responder_rules,
    scenario_hash,
    scenario_id,
    to_json,
    write,
)
from cerl.scenario.generator import GENERATOR_VERSION, generate
from cerl.scenario.responders import (
    EffectKind,
    EffectSpec,
    GuardKind,
    GuardSpec,
    ResponderRule,
    TriggerSpec,
)
from cerl.scenario.schema import (
    ALL_BRANCHES,
    SCHEMA_VERSION,
    BranchSpec,
    FrozenScenario,
    InvariantSpec,
    PermittedDiffSpec,
    RubricItem,
    ScenarioTemplate,
)

__all__ = [
    "ALL_BRANCHES",
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "TEMPLATES",
    "BranchSpec",
    "EffectKind",
    "EffectSpec",
    "FrozenScenario",
    "GuardKind",
    "GuardSpec",
    "InvariantSpec",
    "PermittedDiffSpec",
    "ResponderRule",
    "RubricItem",
    "ScenarioTemplate",
    "TriggerSpec",
    "axes",
    "from_json",
    "generate",
    "load",
    "manifest_entry",
    "materialize",
    "reachable_responder_rules",
    "scenario_hash",
    "scenario_id",
    "to_json",
    "write",
]
