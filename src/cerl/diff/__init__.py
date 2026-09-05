"""Structural diffing, the business projection, and closed-world allowlists.

Operates on plain canonical-JSON documents, so it imports only ``core`` and can
sit beneath ``trace`` and ``state`` without a cycle.
"""

from cerl.diff.allowlist import (
    PermittedDiff,
    ResidualSplit,
    residual,
    residual_of_diffs,
)
from cerl.diff.differ import apply_diff, diff_business, diff_documents
from cerl.diff.models import DiffOp, DiffOpKind, Origin, StateDiff
from cerl.diff.projection import BOOKKEEPING_PATHS, is_bookkeeping, project_business

__all__ = [
    "BOOKKEEPING_PATHS",
    "DiffOp",
    "DiffOpKind",
    "Origin",
    "PermittedDiff",
    "ResidualSplit",
    "StateDiff",
    "apply_diff",
    "diff_business",
    "diff_documents",
    "is_bookkeeping",
    "project_business",
    "residual",
    "residual_of_diffs",
]
