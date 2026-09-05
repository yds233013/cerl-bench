"""The deterministic verifier: predicates, taxonomy, verdict.

``verify`` is a pure, read-only function over state and trace. It imports no
``tools``, ``env``, ``agents`` or ``reference`` (import-linter contract 2).
"""

from cerl.verify.approval import ApprovalCheck, any_usable_approval, check_refund_approval
from cerl.verify.taxonomy import FailureClass, classify
from cerl.verify.verdict import VERIFIER_VERSION, Verdict
from cerl.verify.verifier import declared_outcome, predicate_library_hash, verify

__all__ = [
    "VERIFIER_VERSION",
    "ApprovalCheck",
    "FailureClass",
    "Verdict",
    "any_usable_approval",
    "check_refund_approval",
    "classify",
    "declared_outcome",
    "predicate_library_hash",
    "verify",
]
