"""Decision 1: attempted and committed violations stay separate everywhere.

They are different phenomena. One is realised harm to state; the other is
evidence of unsafe intent that the backend happened to stop. Merging them would
make a policy that repeatedly probes for blocked operations indistinguishable
from one that actually corrupted records, and would silently encode a value
judgement about how much intent counts.

Phase 1 deliberately chooses **no** training weight for attempted costs.
"""

from __future__ import annotations

import ast
import inspect

from cerl.actions import BillingIssueRefund, Finish
from cerl.core import ChargeId
from cerl.env import CerlEnv, CostVector, RewardVector, default_scalar
from cerl.reference import W2Oracle, run_actions, run_reference
from cerl.state import ConstraintClass, ViolationLog
from cerl.trace import TraceEntry
from cerl.verify import Verdict
from tests.helpers import REPO

SRC = REPO / "src" / "cerl"

COMMITTED_TOKENS = ("committed", "violations")
ATTEMPTED_TOKENS = ("attempted",)
MERGING_OPS = (ast.Add, ast.BitOr)
MERGING_CALLS = {"sum", "max", "min", "chain", "union", "update", "extend"}


def _names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id.lower())
        elif isinstance(child, ast.Attribute):
            found.add(child.attr.lower())
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            found.add(child.value.lower())
    return found


def _has(names: set[str], tokens: tuple[str, ...]) -> bool:
    return any(any(t in n for t in tokens) for n in names)


def _mentions_both(left: set[str], right: set[str]) -> bool:
    committed_left = _has(left, COMMITTED_TOKENS) and not _has(left, ATTEMPTED_TOKENS)
    committed_right = _has(right, COMMITTED_TOKENS) and not _has(right, ATTEMPTED_TOKENS)
    return (committed_left and _has(right, ATTEMPTED_TOKENS)) or (
        committed_right and _has(left, ATTEMPTED_TOKENS)
    )


def test_no_committed_attempted_collapse():
    """AST walk: nothing in src/ sums, concatenates or max-es the two series."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, MERGING_OPS):
                if _mentions_both(_names(node.left), _names(node.right)):
                    offenders.append(f"{rel}:{node.lineno} merges the two series")
            elif isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name in MERGING_CALLS and len(node.args) >= 1:
                    joined = _names(node)
                    if _has(joined, COMMITTED_TOKENS) and _has(joined, ATTEMPTED_TOKENS):
                        offenders.append(f"{rel}:{node.lineno} {name}() over both series")
    assert not offenders, "committed/attempted collapse:\n" + "\n".join(offenders)


def test_default_scalar_cannot_receive_attempted_costs():
    """Stronger than documenting a zero weight: make it un-passable."""
    signature = inspect.signature(default_scalar)
    assert "attempted" not in signature.parameters
    assert list(signature.parameters) == ["reward", "committed"]


def test_no_nonzero_default_weight_for_attempted_anywhere():
    text = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    assert "W_ATTEMPTED" not in text
    assert "attempted_weight" not in text


def test_state_holds_two_independent_sequences():
    log = ViolationLog()
    assert log.committed == () and log.attempted == ()
    fields = set(ViolationLog.model_fields)
    assert fields == {"committed", "attempted"}


def test_trace_entry_carries_both_class_tuples_separately():
    fields = TraceEntry.model_fields
    assert "violation_classes" in fields
    assert "attempted_classes" in fields


def test_verdict_exposes_both_separately():
    fields = Verdict.model_fields
    assert "violations" in fields
    assert "attempted_violations" in fields


def test_step_info_reports_two_distinct_cost_vectors(all_frozen):
    scenario = all_frozen[0]
    env = CerlEnv(scenario)
    env.reset()
    result = env.step(Finish(summary=""))
    assert set(result.info["committed_costs"]) == {c.value for c in ConstraintClass}
    assert set(result.info["attempted_costs"]) == {c.value for c in ConstraintClass}
    assert result.info["committed_costs"] is not result.info["attempted_costs"]


def test_a_blocked_attempt_never_becomes_a_committed_violation(all_frozen):
    """The behavioural half: probe an operation the backend refuses."""
    scenario = next(s for s in all_frozen if s.required_decision == "escalate")
    gold = run_reference(scenario, W2Oracle())
    charge = ChargeId(str(scenario.variables["target_charge"]))
    amount = int(scenario.variables["duplicate_amount"])
    probe = BillingIssueRefund(charge_id=charge, amount_cents=amount * 100)

    cut = len(gold.actions) - 1
    episode = run_actions(scenario, (*gold.actions[:cut], probe, *gold.actions[cut:]))
    verdict = episode.verdict

    assert verdict.attempted_violations, "the blocked probe must be recorded"
    assert not verdict.violations, "a blocked attempt must never latch as committed"
    assert verdict.safe_completion, "no state damage means SafeCompletion still holds"
    assert str(verdict.failure_class) == "INTENT_UNSAFE_BLOCKED"


def test_committed_and_attempted_totals_are_reported_apart():
    committed = CostVector.from_classes((ConstraintClass.AUTH,))
    attempted = CostVector.from_classes((ConstraintClass.AMOUNT,))
    assert committed.total == 1
    assert attempted.total == 1
    # The scalar reflects only committed harm.
    scalar = default_scalar(RewardVector(r_outcome=1, r_task=1.0, r_decision=1), committed)
    clean = default_scalar(RewardVector(r_outcome=1, r_task=1.0, r_decision=1), CostVector())
    assert scalar < clean
