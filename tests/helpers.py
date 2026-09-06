"""Shared test helpers reachable as a plain module (not via the conftest package)."""

from __future__ import annotations

from pathlib import Path

from cerl.scenario import freeze as _freeze

REPO = Path(__file__).resolve().parents[1]
#: The canonical corpus (2.0.0). The 1.x corpus stays on disk under
#: ``scenarios/frozen`` as an immutable historical artifact.
FROZEN_DIR = REPO / "scenarios" / "v2" / "frozen"
GOLD_DIR = REPO / "scenarios" / "v2" / "gold"
LEGACY_FROZEN_DIR = REPO / "scenarios" / "frozen"
LEGACY_MANIFEST = REPO / "scenarios" / "manifest.json"


def frozen_paths() -> list[Path]:
    return sorted(FROZEN_DIR.glob("*.json"))


def load_all_frozen():
    return [_freeze.load(p) for p in frozen_paths()]
