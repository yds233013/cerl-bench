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

SPLIT_VERSION = "1.1.0"

#: Provenance for the eligibility layer added in 1.1.0.
#:
#: **1.0.0 -> 1.1.0 (2026-09-05).** An inventory of intervention-axis values by
#: partition found that 85 of the 144 ``train`` scenarios carry a value
#: registered as held out in ``siblings``. That is a direct consequence of
#: partition rule 1 -- a counterfactual and its ID sibling land in the same
#: partition -- and it is correct *for pairing*. It is wrong for *training*: the
#: design's CF-axis tier requires held-out values to be absent from training
#: data, and a partition label alone never encoded that.
#:
#: The two requirements are not reconcilable in one label, so 1.1.0 stops trying.
#: ``Partition`` keeps its approved meaning unchanged, and a separate, explicit
#: **training-eligibility** predicate is layered on top. No scenario moved
#: partitions and no file was regenerated; what changed is that "may be trained
#: on" is now computed rather than assumed from the partition name.
ELIGIBILITY_VERSION = "1.1.0"


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


# --------------------------------------------------------------------------
# training eligibility -- added in split 1.1.0
# --------------------------------------------------------------------------


class Ineligibility(StrEnum):
    """Why a scenario may not be used as training data."""

    NOT_TRAIN_PARTITION = "not_train_partition"
    REGISTERED_HELD_OUT = "registered_held_out"


def training_ineligibility(scenario: FrozenScenario) -> Ineligibility | None:
    """Why this scenario is not training data, or ``None`` if it is.

    Two independent reasons, and they are genuinely independent: a scenario can
    sit in the ``train`` partition and still be a registered counterfactual,
    which is exactly the case the inventory surfaced.

    The partition answers "which side of the paired comparison does this group
    belong to". This answers "may a policy be trained on it". Conflating them
    let 85 counterfactual scenarios read as training data.
    """
    if partition_of(scenario) is not Partition.TRAIN:
        return Ineligibility.NOT_TRAIN_PARTITION
    if siblings.is_held_out(scenario.axes, scenario.template_id):
        return Ineligibility.REGISTERED_HELD_OUT
    return None


def is_training_eligible(scenario: FrozenScenario) -> bool:
    return training_ineligibility(scenario) is None


def training_eligible(scenarios: list[FrozenScenario]) -> list[FrozenScenario]:
    """The scenarios a policy may be trained on, or piloted against."""
    return [s for s in scenarios if is_training_eligible(s)]


class EligibilityInventory(Frozen):
    """Per-family, per-partition inventory of intervention-axis values."""

    version: str = ELIGIBILITY_VERSION
    #: ``family|axis|partition|value -> count``
    counts: FrozenMap[str, int] = FrozenMap()
    #: ``family|axis|value`` entries registered as held out.
    held_out_values: tuple[str, ...] = ()
    #: Held-out values found inside the train partition. Non-empty is expected
    #: and is precisely what the eligibility layer exists to exclude.
    held_out_in_train: FrozenMap[str, int] = FrozenMap()
    eligible: int = 0
    total: int = 0


def inventory(scenarios: list[FrozenScenario]) -> EligibilityInventory:
    """Build the inventory that justifies the eligibility rule."""
    counts: dict[str, int] = {}
    held_out: set[str] = set()
    in_train: dict[str, int] = {}

    for scenario in scenarios:
        pairing = siblings.pairing_for(scenario.template_id)
        partition = partition_of(scenario).value
        for intervention in pairing.interventions:
            value = scenario.axes.get(intervention.axis, "")
            key = f"{scenario.family}|{intervention.axis}|{partition}|{value}"
            counts[key] = counts.get(key, 0) + 1
            if intervention.is_held_out(value):
                held_out.add(f"{scenario.family}|{intervention.axis}|{value}")
                if partition == Partition.TRAIN.value:
                    tag = f"{scenario.family}|{intervention.axis}|{value}"
                    in_train[tag] = in_train.get(tag, 0) + 1

    return EligibilityInventory(
        counts=FrozenMap(counts),
        held_out_values=tuple(sorted(held_out)),
        held_out_in_train=FrozenMap(in_train),
        eligible=len(training_eligible(scenarios)),
        total=len(scenarios),
    )
