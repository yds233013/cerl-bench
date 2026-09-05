"""PRIVILEGED: ground truth, oracle policies, gold trajectories, mutations.

Nothing in ``env``, ``tools`` or ``agents`` may import this package
(import-linter contracts 1 and 3).
"""

from cerl.reference.gold import GoldTrajectory, produce
from cerl.reference.ground_truth import GroundTruthView, ReferencePolicy, ground_truth_for
from cerl.reference.oracle.w2_refund import W2Oracle
from cerl.reference.runner import Episode, run_actions, run_reference
from cerl.reference.variants import (
    DifferenceReport,
    W2AlternativePolicy,
    compare,
)

__all__ = [
    "DifferenceReport",
    "Episode",
    "GoldTrajectory",
    "GroundTruthView",
    "ReferencePolicy",
    "W2AlternativePolicy",
    "W2Oracle",
    "compare",
    "ground_truth_for",
    "produce",
    "run_actions",
    "run_reference",
]
