"""PRIVILEGED: ground truth, oracle policies, gold trajectories, mutations.

Nothing in ``env``, ``tools`` or ``agents`` may import this package
(import-linter contracts 1 and 3).
"""

from cerl.reference.gold import GoldTrajectory, produce
from cerl.reference.ground_truth import GroundTruthView, ReferencePolicy, ground_truth_for
from cerl.reference.oracle.w1_profile import W1Oracle
from cerl.reference.oracle.w2_refund import W2Oracle
from cerl.reference.oracle.w3_fraud import W3Oracle
from cerl.reference.registry import alternative_for, families, oracle_for
from cerl.reference.runner import Episode, run_actions, run_reference
from cerl.reference.variants import (
    DifferenceReport,
    W1AlternativePolicy,
    W2AlternativePolicy,
    W3AlternativePolicy,
    compare,
)

__all__ = [
    "DifferenceReport",
    "Episode",
    "GoldTrajectory",
    "GroundTruthView",
    "ReferencePolicy",
    "W1AlternativePolicy",
    "W1Oracle",
    "W2AlternativePolicy",
    "W2Oracle",
    "W3AlternativePolicy",
    "W3Oracle",
    "alternative_for",
    "compare",
    "families",
    "ground_truth_for",
    "oracle_for",
    "produce",
    "run_actions",
    "run_reference",
]
