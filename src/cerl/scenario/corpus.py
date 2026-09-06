"""Corpus versions and where each one's files live.

A corpus version is the unit that changes when *file contents* change. Splits
version independently, because relabelling which partition a group belongs to
does not touch a single byte of a scenario.

======  ==========================================================
1.0.0   Generated before partitions existed. Every scenario draws from the
        ``core`` lexicon pool regardless of partition, so Criterion 42's
        lexicon clause fails and cannot be repaired by relabelling. **Kept
        immutable.** Files, manifest and split manifest are untouched.
2.0.0   Partition decided from the plan, then generation draws from that
        partition's shard. Lexicon values are disjoint across train,
        validation and evaluation in the committed files. **Canonical.**
======  ==========================================================

Hashes changed between them because the files changed: different names produce
different bytes. That is the point of the bump rather than a cost of it -- a
regenerated corpus that kept its hashes would mean nothing had actually been
fixed.
"""

from __future__ import annotations

from pathlib import Path

from cerl.core import Frozen

CORPUS_VERSION = "2.0.0"
LEGACY_CORPUS_VERSION = "1.0.0"


class CorpusPaths(Frozen):
    """Where one corpus version's committed artifacts live."""

    version: str
    frozen: Path
    gold: Path
    manifest: Path
    split_manifest: Path
    #: True for the version consumers should read.
    canonical: bool = False

    model_config = Frozen.model_config | {"arbitrary_types_allowed": True}


LEGACY = CorpusPaths(
    version=LEGACY_CORPUS_VERSION,
    frozen=Path("scenarios/frozen"),
    gold=Path("scenarios/gold"),
    manifest=Path("scenarios/manifest.json"),
    split_manifest=Path("scenarios/split_manifest.json"),
)

CANONICAL = CorpusPaths(
    version=CORPUS_VERSION,
    frozen=Path("scenarios/v2/frozen"),
    gold=Path("scenarios/v2/gold"),
    manifest=Path("scenarios/v2/manifest.json"),
    split_manifest=Path("scenarios/v2/split_manifest.json"),
    canonical=True,
)

VERSIONS: tuple[CorpusPaths, ...] = (LEGACY, CANONICAL)


def paths_for(version: str) -> CorpusPaths:
    for entry in VERSIONS:
        if entry.version == version:
            return entry
    raise KeyError(f"unknown corpus version {version!r}")


def canonical() -> CorpusPaths:
    return CANONICAL
