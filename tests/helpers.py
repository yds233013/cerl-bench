"""Shared test helpers reachable as a plain module (not via the conftest package)."""

from __future__ import annotations

from pathlib import Path

from cerl.scenario import freeze as _freeze

REPO = Path(__file__).resolve().parents[1]
FROZEN_DIR = REPO / "scenarios" / "frozen"
GOLD_DIR = REPO / "scenarios" / "gold"


def frozen_paths() -> list[Path]:
    return sorted(FROZEN_DIR.glob("*.json"))


def load_all_frozen():
    return [_freeze.load(p) for p in frozen_paths()]
