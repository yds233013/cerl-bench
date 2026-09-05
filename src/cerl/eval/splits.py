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

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

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

#: The canonical split. See ``docs/pilot-split-audit.md`` for the migration.
#:
#: **1.1.0 -> 1.2.0 (2026-09-05).** 1.1.0 added a runtime *eligibility filter*
#: that excluded registered holdouts at selection time. That kept the pilot
#: clean but left the canonical split itself violating Criterion 42: the
#: training partition still contained 85 held-out scenarios, and anything
#: reading ``partition_of`` directly -- a future training loop included -- would
#: have consumed them. A filter in one consumer is not a fixed split.
#:
#: 1.2.0 repairs the split itself. The rule: **a sibling group containing any
#: registered held-out value is never training data.** Such groups move whole,
#: to validation or evaluation, so pair integrity is preserved. Pure-ID groups
#: keep their 1.0.0 assignment exactly.
#:
#: Nothing is regenerated. Only the partition label of complete sibling groups
#: changes; every scenario file, hash, seed, generator version and provenance
#: field is untouched.
SPLIT_VERSION_CANONICAL = "1.2.0"

#: Of the CF-bearing groups that must leave training, this fraction goes to
#: validation and the rest to evaluation, preserving the 15:25 ratio 1.0.0 used.
CF_VALIDATION_SHARE = 38


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


def partition_v1_0_0(scenario: FrozenScenario) -> Partition:
    """The 1.0.0 assignment. **Frozen historical artifact -- do not change.**

    Kept verbatim so results recorded against 1.0.0 stay interpretable and so
    the 1.2.0 migration can state exactly which groups moved. It is not the
    canonical split any more: it satisfies pair integrity but not Criterion 42,
    because a counterfactual and its ID sibling share a partition and 60% of
    pair groups land in train.
    """
    bucket = int(pair_key(scenario)[:8], 16) % 100
    if bucket < TRAIN_SHARE:
        return Partition.TRAIN
    if bucket < TRAIN_SHARE + VALIDATION_SHARE:
        return Partition.VALIDATION
    return Partition.EVALUATION


def partition_of(scenario: FrozenScenario) -> Partition:
    """The canonical partition, split 1.2.0.

    Reads the committed manifest, because the 1.2.0 rule is a function of the
    whole corpus rather than of one scenario: whether a sibling group may be
    training data depends on whether *any* member of that group carries a
    registered held-out value, which a single scenario cannot know.
    """
    return manifest().partition_of(scenario)


class SplitSummary(Frozen):
    counts: FrozenMap[str, int]
    version: str = SPLIT_VERSION_CANONICAL

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

    Under the canonical 1.2.0 split the second clause is **redundant**: the
    training partition contains no registered holdout, so no scenario can be
    excluded for that reason. It is kept as a live assertion rather than deleted
    -- if a future split regression puts a counterfactual back into training,
    this catches it at every consumer instead of only where someone remembered
    to check. Under 1.1.0 it was the whole defence; now it is the backstop.
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

    version: str = SPLIT_VERSION_CANONICAL
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


# --------------------------------------------------------------------------
# the canonical split manifest -- version 1.2.0
# --------------------------------------------------------------------------

MANIFEST_PATH = Path("scenarios/split_manifest.json")


class GroupAssignment(Frozen):
    """One sibling group's canonical partition, with why it is there."""

    pair_key: str
    partition: Partition
    #: Scenario ids in the group, sorted. Every one moves together or not at all.
    members: tuple[str, ...]
    #: True when any member carries a registered held-out value.
    cf_bearing: bool
    partition_v1_0_0: Partition
    reason: str

    @property
    def moved(self) -> bool:
        return self.partition is not self.partition_v1_0_0


class SplitManifest(Frozen):
    """The committed canonical split.

    A file rather than a function because the 1.2.0 rule needs the whole corpus:
    whether a group may be training data depends on whether any of its members
    is a registered counterfactual. Committing it also makes the migration
    auditable -- ``moved_groups`` names exactly what changed and why.
    """

    version: str = SPLIT_VERSION_CANONICAL
    groups: tuple[GroupAssignment, ...] = ()

    def partition_of(self, scenario: FrozenScenario) -> Partition:
        key = pair_key(scenario)
        for group in self.groups:
            if group.pair_key == key:
                return group.partition
        raise KeyError(
            f"{scenario.scenario_id} is in no manifest group; the manifest is "
            f"stale. Rebuild with `cerl splits --write`.",
        )

    @property
    def moved_groups(self) -> tuple[GroupAssignment, ...]:
        return tuple(g for g in self.groups if g.moved)

    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(m for g in self.groups for m in g.members))


def _held_out_labels(group: list[FrozenScenario]) -> set[str]:
    """``family/axis=value`` for each registered holdout in the group.

    Namespaced by family and axis on purpose: ``signal_count=1`` and
    ``approval=expired`` are unrelated facts about different workflows, and a
    bare value would collide across families the moment two share an axis name.
    """
    labels: set[str] = set()
    for scenario in group:
        pairing = siblings.pairing_for(scenario.template_id)
        intervention = pairing.intervention_for(scenario.axes)
        if intervention is None:
            continue
        value = scenario.axes[intervention.axis]
        labels.add(f"{scenario.family}/{intervention.axis}={value}")
    return labels


def build_manifest(scenarios: list[FrozenScenario]) -> SplitManifest:
    """Derive the canonical split from the corpus. Deterministic and total.

    Total is the operative word: every scenario lands in exactly one group and
    every group in exactly one partition, so nothing can be silently dropped or
    counted twice. ``verify_manifest_totality`` checks that against the corpus.
    """
    members: dict[str, list[FrozenScenario]] = {}
    for scenario in scenarios:
        members.setdefault(pair_key(scenario), []).append(scenario)

    assignments: list[GroupAssignment] = []
    for key in sorted(members):
        group = members[key]
        cf_bearing = any(
            siblings.is_held_out(s.axes, s.template_id) for s in group
        )
        original = partition_v1_0_0(group[0])

        if not cf_bearing:
            partition = original
            reason = "pure-ID group; 1.0.0 assignment preserved"
        elif original is not Partition.TRAIN:
            partition = original
            reason = "CF-bearing group already outside training; unchanged"
        else:
            # Must leave training, and must leave whole.
            bucket = int(key[8:16], 16) % 100
            partition = (
                Partition.VALIDATION
                if bucket < CF_VALIDATION_SHARE
                else Partition.EVALUATION
            )
            held = sorted(_held_out_labels(group))
            reason = (
                f"moved out of training: group carries registered held-out "
                f"{', '.join(held)}"
            )

        assignments.append(
            GroupAssignment(
                pair_key=key,
                partition=partition,
                members=tuple(sorted(s.scenario_id for s in group)),
                cf_bearing=cf_bearing,
                partition_v1_0_0=original,
                reason=reason,
            ),
        )
    return SplitManifest(groups=tuple(assignments))


def write_manifest(manifest: SplitManifest, path: Path = MANIFEST_PATH) -> Path:
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = content_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> SplitManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("manifest_hash", None)
    return SplitManifest.model_validate(payload)


@lru_cache(maxsize=1)
def manifest() -> SplitManifest:
    """The committed canonical split, loaded once."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"no canonical split manifest at {MANIFEST_PATH}; "
            f"build it with `cerl splits --write`",
        )
    return load_manifest()


def verify_manifest_totality(
    scenarios: list[FrozenScenario], split: SplitManifest,
) -> tuple[str, ...]:
    """Nothing omitted, relabelled, or duplicated. Returns the problems found."""
    problems: list[str] = []
    corpus = sorted(s.scenario_id for s in scenarios)
    listed = list(split.scenario_ids())

    missing = set(corpus) - set(listed)
    extra = set(listed) - set(corpus)
    problems.extend(f"omitted from the manifest: {sid}" for sid in sorted(missing))
    problems.extend(f"in the manifest but not the corpus: {sid}" for sid in sorted(extra))
    if len(listed) != len(set(listed)):
        duplicated = {sid for sid in listed if listed.count(sid) > 1}
        problems.extend(f"listed more than once: {sid}" for sid in sorted(duplicated))

    problems.extend(
        f"empty group {group.pair_key}" for group in split.groups if not group.members
    )
    return tuple(problems)
