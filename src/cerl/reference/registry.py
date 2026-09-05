"""Family -> reference policy mapping.

Lives in ``reference`` rather than ``scenario`` because ``scenario`` may not
import ``reference`` (import-linter contract 4): scenario data is inert and must
not be able to reach the privileged machinery.

Each family declares two policies. The **oracle** is the registered reference:
it defines ``oracle_tool_calls`` and therefore the difficulty metric used by
Criterion 41. The **alternative** is a second, materially different correct
policy, used to prove the rubric and allowlist are not so tight that only one
trajectory passes.
"""

from __future__ import annotations

from collections.abc import Callable

from cerl.reference.ground_truth import ReferencePolicy
from cerl.reference.oracle.w1_profile import W1Oracle
from cerl.reference.oracle.w2_refund import W2Oracle
from cerl.reference.oracle.w3_fraud import W3Oracle
from cerl.reference.variants import (
    W1AlternativePolicy,
    W2AlternativePolicy,
    W3AlternativePolicy,
)
from cerl.scenario.schema import FrozenScenario

PolicyFactory = Callable[[], ReferencePolicy]

W1_FAMILY = "duplicate_billing_profile"
W2_FAMILY = "duplicate_charge_approval"
W3_FAMILY = "suspicious_refund_escalation"

#: The policy whose tool-call count defines the registered difficulty metric.
ORACLES: dict[str, PolicyFactory] = {
    W1_FAMILY: W1Oracle,
    W2_FAMILY: W2Oracle,
    W3_FAMILY: W3Oracle,
}

#: A second, materially different correct policy per family.
ALTERNATIVES: dict[str, PolicyFactory] = {
    W1_FAMILY: W1AlternativePolicy,
    W2_FAMILY: W2AlternativePolicy,
    W3_FAMILY: W3AlternativePolicy,
}


def oracle_for(scenario: FrozenScenario) -> ReferencePolicy:
    if scenario.family not in ORACLES:
        raise KeyError(f"no oracle registered for family {scenario.family!r}")
    return ORACLES[scenario.family]()


def alternative_for(scenario: FrozenScenario) -> ReferencePolicy:
    if scenario.family not in ALTERNATIVES:
        raise KeyError(f"no alternative policy registered for family {scenario.family!r}")
    return ALTERNATIVES[scenario.family]()


def families() -> tuple[str, ...]:
    return tuple(sorted(ORACLES))
