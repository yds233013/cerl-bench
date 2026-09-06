"""Partition assignment, computed **before** a world is materialised.

Corpus 1.x decided partitions after generation, from the finished scenarios.
That ordering is why every file drew from one lexicon pool: by the time anything
knew which partition a scenario belonged to, its names had already been chosen.

Here the assignment is a function of the *plan* -- template, axis assignment and
seed -- all of which exist before any entity does. The generator can therefore
be handed the right shard, and lexicon disjointness becomes a property of
generation rather than something to check for afterwards and be unable to fix.

This lives in ``scenario`` rather than ``eval`` because a split is a property of
the corpus, not of a consumer of it. ``eval.splits`` re-exports the public names.
"""

from __future__ import annotations

from enum import StrEnum

from cerl.core import Frozen, FrozenMap, content_hash
from cerl.scenario import siblings


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


#: Fraction of pure-ID groups assigned to each partition, in units of 1/100.
#: Unchanged from split 1.0.0, so a pure-ID group keeps its historical home.
TRAIN_SHARE = 60
VALIDATION_SHARE = 15

#: Of the CF-bearing groups, which may never be training data, this fraction
#: goes to validation and the rest to evaluation -- the 15:25 ratio 1.0.0 used.
CF_VALIDATION_SHARE = 38


def group_key(template_id: str, axes: FrozenMap[str, str], seed: int) -> str:
    """The key shared by a counterfactual and its in-distribution sibling.

    Both members reduce to the *sibling* assignment, so they hash to the same
    group and therefore always land in the same partition.
    """
    resolved = axes
    if siblings.is_held_out(axes, template_id):
        resolved = siblings.sibling_axes(axes, template_id)
    return content_hash(
        {"template": template_id, "axes": dict(resolved), "seed": seed},
    )


def partition_v1_0_0_for_key(key: str) -> Partition:
    """The 1.0.0 rule. **Frozen historical artifact -- do not change.**"""
    bucket = int(key[:8], 16) % 100
    if bucket < TRAIN_SHARE:
        return Partition.TRAIN
    if bucket < TRAIN_SHARE + VALIDATION_SHARE:
        return Partition.VALIDATION
    return Partition.EVALUATION


class PlanEntry(Frozen):
    """One scenario as it exists before generation."""

    template_id: str
    family: str
    axes: FrozenMap[str, str]
    seed: int

    @property
    def key(self) -> str:
        return group_key(self.template_id, self.axes, self.seed)

    @property
    def held_out(self) -> bool:
        return siblings.is_held_out(self.axes, self.template_id)


class GroupPlan(Frozen):
    """A sibling group and the partition its whole membership will occupy."""

    key: str
    partition: Partition
    partition_v1_0_0: Partition
    cf_bearing: bool
    reason: str
    held_out_labels: tuple[str, ...] = ()

    @property
    def moved(self) -> bool:
        return self.partition is not self.partition_v1_0_0


def _held_out_labels(entries: list[PlanEntry]) -> tuple[str, ...]:
    """``family/axis=value`` per registered holdout in the group.

    Namespaced by family and axis deliberately: W1's ``merge_approval`` and W2's
    ``approval`` share the value names ``expired`` and ``missing_unobtainable``,
    so a bare value would conflate two different workflows' counterfactuals.
    """
    labels: set[str] = set()
    for entry in entries:
        pairing = siblings.pairing_for(entry.template_id)
        intervention = pairing.intervention_for(entry.axes)
        if intervention is None:
            continue
        labels.add(
            f"{entry.family}/{intervention.axis}={entry.axes[intervention.axis]}",
        )
    return tuple(sorted(labels))


def assign(entries: list[PlanEntry]) -> dict[str, GroupPlan]:
    """Assign every sibling group to a partition. Deterministic and total.

    The rule, unchanged from split 1.2.0: **a group containing any registered
    held-out value is never training data.** Such groups move whole, so a
    counterfactual and its sibling are never separated. Pure-ID groups keep
    their 1.0.0 assignment exactly.
    """
    grouped: dict[str, list[PlanEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.key, []).append(entry)

    plans: dict[str, GroupPlan] = {}
    for key in sorted(grouped):
        members = grouped[key]
        cf_bearing = any(m.held_out for m in members)
        original = partition_v1_0_0_for_key(key)

        if not cf_bearing:
            partition = original
            reason = "pure-ID group; 1.0.0 assignment preserved"
        elif original is not Partition.TRAIN:
            partition = original
            reason = "CF-bearing group already outside training; unchanged"
        else:
            bucket = int(key[8:16], 16) % 100
            partition = (
                Partition.VALIDATION
                if bucket < CF_VALIDATION_SHARE
                else Partition.EVALUATION
            )
            reason = "moved out of training: group carries a registered holdout"

        plans[key] = GroupPlan(
            key=key,
            partition=partition,
            partition_v1_0_0=original,
            cf_bearing=cf_bearing,
            reason=reason,
            held_out_labels=_held_out_labels(members),
        )
    return plans


def shard_for(partition: Partition) -> str:
    """The lexicon shard a partition draws from. One-to-one, by construction."""
    return partition.value
