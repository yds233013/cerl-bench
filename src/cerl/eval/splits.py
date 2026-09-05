"""Training, validation and evaluation partitions.

Two rules, both structural rather than advisory:

1. **A counterfactual and its ID sibling land in the same partition.** The
   ID/CF contrast is the measurement; splitting a pair across partitions would
   put half of a comparison in training and make the gap uninterpretable.
2. **A pair's shared entities never straddle the split.** Both members of a pair
   are generated from the same seed and therefore contain the *same* customers
   with the same names. If one member were in training, an agent could recognise
   the evaluation world by its entities rather than by reasoning about it.

Assignment is deterministic: it hashes the pair's identity, never a wall clock
or an iteration order, so the partition a scenario belongs to is a stable
property of the corpus rather than of the run that computed it.
"""

from __future__ import annotations

from enum import StrEnum

from cerl.core import Frozen, FrozenMap, content_hash
from cerl.scenario import siblings
from cerl.scenario.schema import FrozenScenario


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


#: Fraction of pair-groups assigned to each partition, in units of 1/100.
TRAIN_SHARE = 60
VALIDATION_SHARE = 15
# The remainder is evaluation.

SPLIT_VERSION = "1.0.0"


def pair_key(scenario: FrozenScenario) -> str:
    """The grouping key shared by a counterfactual and its ID sibling.

    Both members reduce to the *sibling* assignment, so they always hash to the
    same group and therefore the same partition.
    """
    axes = scenario.axes
    if siblings.is_held_out(axes, scenario.template_id):
        axes = siblings.sibling_axes(axes, scenario.template_id)
    return content_hash(
        {
            "template": scenario.template_id,
            "axes": dict(axes),
            "seed": scenario.root_seed,
        },
    )


def partition_of(scenario: FrozenScenario) -> Partition:
    """Deterministic partition for one scenario."""
    bucket = int(pair_key(scenario)[:8], 16) % 100
    if bucket < TRAIN_SHARE:
        return Partition.TRAIN
    if bucket < TRAIN_SHARE + VALIDATION_SHARE:
        return Partition.VALIDATION
    return Partition.EVALUATION


class SplitSummary(Frozen):
    counts: FrozenMap[str, int]
    version: str = SPLIT_VERSION

    def total(self) -> int:
        return sum(self.counts.values())


def summarize(scenarios: list[FrozenScenario]) -> SplitSummary:
    counts: dict[str, int] = {p.value: 0 for p in Partition}
    for scenario in scenarios:
        counts[partition_of(scenario).value] += 1
    return SplitSummary(counts=FrozenMap(counts))


def select(scenarios: list[FrozenScenario], partition: Partition | None) -> list[FrozenScenario]:
    if partition is None:
        return list(scenarios)
    return [s for s in scenarios if partition_of(s) is partition]
