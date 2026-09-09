"""Phase 2 local RL pilot.

Deliberately a **separate top-level package**, not part of ``cerl``. The
benchmark must remain installable, importable and testable with no training
dependency present, and CI must keep running offline exactly as it does today.
Nothing in ``cerl`` imports this package; everything here imports ``cerl`` as an
ordinary consumer would.
"""
