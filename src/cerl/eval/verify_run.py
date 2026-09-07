"""Offline verification of a recorded run.

Given a manifest, replay its recorded actions through the frozen scenarios and
check that *everything it claims* reproduces: state hashes, responder
transitions and their logical times, chain hashes, verdicts and aggregate
metrics. No model, no network, no API key.

The point is that verification compares against **recorded evidence** rather
than accepting freshly regenerated output as ground truth. A check that simply
re-ran the pipeline and compared it with itself would pass on a corrupted
manifest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from cerl.core import Frozen
from cerl.eval import manifest as manifest_module
from cerl.eval.manifest import EpisodeRecord, RunManifest, current_versions
from cerl.eval.metrics import aggregate
from cerl.reference.runner import run_actions
from cerl.scenario import freeze as freeze_module
from cerl.scenario.freeze import scenario_hash
from cerl.scenario.schema import FrozenScenario


class Mismatch(Frozen):
    scenario_id: str
    field: str
    recorded: str
    recomputed: str

    def describe(self) -> str:
        return (
            f"{self.scenario_id}: {self.field} recorded={self.recorded} "
            f"recomputed={self.recomputed}"
        )


class VerificationReport(Frozen):
    episodes_checked: int
    mismatches: tuple[Mismatch, ...] = ()
    version_skew: tuple[str, ...] = ()
    missing_scenarios: tuple[str, ...] = ()
    #: The manifest's own checksum, checked separately from replay. Absent on
    #: historical files, which is reported rather than failed.
    checksum: str = "not checked"
    checksum_ok: bool = True
    #: Claims this verification does not and cannot support.
    unverifiable: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (
            self.mismatches
            or self.version_skew
            or self.missing_scenarios
            or not self.checksum_ok
        )

    def summary(self) -> str:
        if self.ok and self.episodes_checked == 0:
            # An empty manifest passes every check there is to run, which is not
            # the same as being evidence of anything. Interrupted runs produce
            # these, so the distinction is real and worth stating rather than
            # letting a bare "VERIFIED" imply a result was reproduced.
            return (
                "manifest is well-formed and its checksum is valid, but it "
                "records NO episodes; there is nothing to reproduce"
            )
        if self.ok:
            return f"verified {self.episodes_checked} episodes offline; all evidence matches"
        parts = [f"checked {self.episodes_checked} episodes"]
        if self.version_skew:
            parts.append(f"version skew: {list(self.version_skew)}")
        if self.missing_scenarios:
            parts.append(f"missing scenarios: {list(self.missing_scenarios)[:5]}")
        if not self.checksum_ok:
            parts.append(f"checksum: {self.checksum}")
        if self.mismatches:
            parts.append(
                "mismatches:\n  " + "\n  ".join(m.describe() for m in self.mismatches[:10]),
            )
        return "; ".join(parts)


def _check_versions(manifest: RunManifest) -> tuple[str, ...]:
    """A result graded under different logic is never silently compared."""
    current = current_versions()
    skew = []
    for field, value in current.items():
        recorded = getattr(manifest, field, "")
        if recorded and recorded != value:
            skew.append(f"{field}: manifest={recorded[:12]} current={value[:12]}")
    return tuple(skew)


#: Fields a replay genuinely cannot establish. Empty, and deliberately so: every
#: field of ``EpisodeRecord`` is derived from the scenario and the recorded
#: actions, so replay can rebuild all of them. Producer identity and model
#: provenance -- who ran this, and with which model -- live on ``AgentConfig``
#: and are *not* verifiable by replay; they are separate claims and are reported
#: as unverifiable rather than checked.
UNVERIFIABLE_EPISODE_FIELDS: frozenset[str] = frozenset()

#: Claims replay cannot support, recorded so a reader is not misled into
#: thinking a passing verification vouches for them.
UNVERIFIABLE_CLAIMS: tuple[str, ...] = (
    "who produced this run (agent.provider, agent.name)",
    "which model produced it, and that a model produced it at all",
    "the wall-clock cost or spend attributed to it",
)


def _rebuild(record: EpisodeRecord, scenario: FrozenScenario) -> EpisodeRecord:
    """The record replay says this episode *should* have produced."""
    from cerl.eval.runner import record_episode

    return record_episode(scenario, run_actions(scenario, record.actions))


def _compare(
    record: EpisodeRecord, scenario: FrozenScenario,
) -> tuple[list[Mismatch], EpisodeRecord | None]:
    """Compare **every** replay-derivable field, not a chosen subset.

    Built by reconstructing the record from replay and diffing it whole. An
    earlier version compared a hand-picked list, so ``task_completion``,
    ``correct_final_state``, ``decision_correct`` and most per-step fields were
    never checked: a manifest could claim a perfect score over honest actions
    and verification would report that all evidence matched.

    Comparing the reconstructed record instead means a field added to
    ``EpisodeRecord`` later is checked by default rather than forgotten.

    Returns the mismatches and the rebuilt record, so the caller can aggregate
    over replayed records without paying for a second replay. ``None`` means the
    scenario itself did not match and no replay was attempted.
    """
    mismatches: list[Mismatch] = []

    def note(field: str, recorded: object, recomputed: object) -> None:
        mismatches.append(
            Mismatch(
                scenario_id=record.scenario_id,
                field=field,
                recorded=_short(recorded),
                recomputed=_short(recomputed),
            ),
        )

    # Scenario identity first: replaying against a different world would compare
    # the record with something it never claimed.
    current_hash = scenario_hash(scenario)
    if record.scenario_hash != current_hash:
        note("scenario_hash", record.scenario_hash[:12], current_hash[:12])
        return mismatches, None
    if record.initial_state_hash != scenario.world.state_hash():
        note("initial_state_hash", record.initial_state_hash[:12],
             scenario.world.state_hash()[:12])
        return mismatches, None

    rebuilt = _rebuild(record, scenario)
    recorded_doc = json.loads(record.model_dump_json())
    replayed_doc = json.loads(rebuilt.model_dump_json())

    for field in sorted(set(recorded_doc) | set(replayed_doc)):
        if field in UNVERIFIABLE_EPISODE_FIELDS or field == "steps":
            continue
        if recorded_doc.get(field) != replayed_doc.get(field):
            note(field, recorded_doc.get(field), replayed_doc.get(field))

    # Steps are compared element-wise so a mismatch names the entry and field
    # rather than dumping two long arrays at the reader.
    recorded_steps = recorded_doc.get("steps") or []
    replayed_steps = replayed_doc.get("steps") or []
    if len(recorded_steps) != len(replayed_steps):
        note("step_count", len(recorded_steps), len(replayed_steps))
        return mismatches, rebuilt
    for index, (was, now) in enumerate(zip(recorded_steps, replayed_steps, strict=True)):
        for key in sorted(set(was) | set(now)):
            if was.get(key) != now.get(key):
                note(f"step[{index}].{key}", was.get(key), now.get(key))
    return mismatches, rebuilt


#: Mismatch values are truncated so a report stays readable; the field name is
#: what a reader needs, not a full array dump.
_MAX_REPORTED = 80


def _short(value: object) -> str:
    text = str(value)
    return text if len(text) <= _MAX_REPORTED else text[: _MAX_REPORTED - 3] + "..."


#: Metric keys a replay genuinely cannot recompute, and which are therefore
#: exempt from the exact comparison below. Each entry needs a reason, because
#: every exemption is a place a number can be asserted without being checked.
#:
#: * ``spend_cents`` / ``tokens_*`` -- what a provider billed. Replay runs no
#:   model, so there is nothing to recompute; these describe an event outside
#:   the simulation.
#: * ``wall_clock_seconds`` -- how long the machine took. Not state, not
#:   hashed, and different on every machine by design.
#:
#: The list is deliberately short and deliberately explicit. Adding to it means
#: deciding that a reported number will never be checked again, which is a
#: design decision and not a convenience.
UNVERIFIABLE_METRICS: frozenset[str] = frozenset(
    {
        "spend_cents",
        "tokens_in",
        "tokens_out",
        "wall_clock_seconds",
    },
)


def _compare_metrics(
    manifest: RunManifest, replayed: Sequence[EpisodeRecord],
) -> list[Mismatch]:
    """Compare the **complete** metric block, key set included.

    Three failures are possible and all three must be caught:

    * a **changed** value -- the obvious one;
    * a **missing** key, which an earlier version treated as "nothing to
      compare" and passed, so deleting an inconvenient metric hid it;
    * an **extra** key, which nothing looked at, so a manifest could report an
      invented metric beside honest ones and have it pass unchallenged.

    Recomputation is from the **replayed** records, never the submitted ones:
    deriving aggregates from the submission proves only that the file is
    internally consistent, which a submitter can arrange for any numbers.

    ``committed_*`` and ``attempted_*`` are compared as the separate keys they
    are. Nothing here merges them, and a mismatch names whichever series moved.
    """
    recomputed = aggregate(list(replayed))
    recorded = {str(k): v for k, v in manifest.metrics.items()}

    out: list[Mismatch] = []

    def note(field: str, was: object, now: object) -> None:
        out.append(
            Mismatch(
                scenario_id="<aggregate>",
                field=f"metrics.{field}",
                recorded=_short(was),
                recomputed=_short(now),
            ),
        )

    for key in sorted(set(recorded) | set(recomputed)):
        if key in UNVERIFIABLE_METRICS:
            continue
        if key not in recorded:
            note(key, "<absent from the manifest>", recomputed[key])
        elif key not in recomputed:
            note(key, recorded[key], "<not a metric this run produces>")
        elif recorded[key] != recomputed[key]:
            note(key, recorded[key], recomputed[key])
    return out


def verify_manifest(
    manifest: RunManifest,
    frozen_dir: Path,
    manifest_path: Path | None = None,
) -> VerificationReport:
    """Replay a recorded run offline and compare against its own evidence.

    ``manifest_path`` additionally validates the file's own checksum. That is a
    separate and weaker claim -- it catches corruption, not dishonesty, since a
    submitter who edits the contents can recompute it -- so it is reported
    apart from the replay result.
    """
    mismatches: list[Mismatch] = []
    missing: list[str] = []
    checked = 0

    replayed_records: list[EpisodeRecord] = []
    for record in manifest.episodes:
        path = frozen_dir / f"{record.scenario_id}.json"
        if not path.exists():
            missing.append(record.scenario_id)
            continue
        scenario = freeze_module.load(path)
        episode_mismatches, rebuilt = _compare(record, scenario)
        mismatches += episode_mismatches
        if rebuilt is not None:
            replayed_records.append(rebuilt)
        checked += 1

    mismatches += _compare_metrics(manifest, replayed_records)

    checksum = (
        manifest_module.checksum_of(manifest_path)
        if manifest_path is not None
        else None
    )
    return VerificationReport(
        checksum="not checked" if checksum is None else checksum.status,
        checksum_ok=True if checksum is None else checksum.valid,
        unverifiable=UNVERIFIABLE_CLAIMS,
        episodes_checked=checked,
        mismatches=tuple(mismatches),
        version_skew=_check_versions(manifest),
        missing_scenarios=tuple(missing),
    )
