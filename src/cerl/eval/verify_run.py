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

from pathlib import Path

from cerl.core import Frozen
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

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.version_skew or self.missing_scenarios)

    def summary(self) -> str:
        if self.ok:
            return f"verified {self.episodes_checked} episodes offline; all evidence matches"
        parts = [f"checked {self.episodes_checked} episodes"]
        if self.version_skew:
            parts.append(f"version skew: {list(self.version_skew)}")
        if self.missing_scenarios:
            parts.append(f"missing scenarios: {list(self.missing_scenarios)[:5]}")
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


def _compare(record: EpisodeRecord, scenario: FrozenScenario) -> list[Mismatch]:
    mismatches: list[Mismatch] = []

    def note(field: str, recorded: object, recomputed: object) -> None:
        mismatches.append(
            Mismatch(
                scenario_id=record.scenario_id,
                field=field,
                recorded=str(recorded),
                recomputed=str(recomputed),
            ),
        )

    # Scenario identity, and the identity of the world it produced. A file could
    # be edited without its id changing.
    current_hash = scenario_hash(scenario)
    if record.scenario_hash != current_hash:
        note("scenario_hash", record.scenario_hash[:12], current_hash[:12])
        return mismatches
    if record.initial_state_hash != scenario.world.state_hash():
        note("initial_state_hash", record.initial_state_hash[:12],
             scenario.world.state_hash()[:12])
        return mismatches

    episode = run_actions(scenario, record.actions)
    verdict = episode.verdict

    if episode.final.state_hash() != record.terminal_state_hash:
        note("terminal_state_hash", record.terminal_state_hash[:12],
             episode.final.state_hash()[:12])
    if episode.trace.head_hash != record.trace_head_hash:
        note("trace_head_hash", record.trace_head_hash[:12], episode.trace.head_hash[:12])
    if str(verdict.failure_class) != str(record.failure_class):
        note("failure_class", record.failure_class, verdict.failure_class)
    if dict(verdict.rubric) != dict(record.rubric):
        note("rubric", dict(record.rubric), dict(verdict.rubric))
    if verdict.safe_completion != record.safe_completion:
        note("safe_completion", record.safe_completion, verdict.safe_completion)
    if len(verdict.violations) != len(record.committed_violations):
        note("committed_violations", len(record.committed_violations),
             len(verdict.violations))
    if len(verdict.attempted_violations) != len(record.attempted_violations):
        note("attempted_violations", len(record.attempted_violations),
             len(verdict.attempted_violations))
    if verdict.tool_calls != record.tool_calls:
        note("tool_calls", record.tool_calls, verdict.tool_calls)

    # Step-level evidence: every entry hash, logical time and responder rule.
    replayed = episode.trace.entries
    if len(replayed) != len(record.steps):
        note("step_count", len(record.steps), len(replayed))
        return mismatches
    for recorded_step, entry in zip(record.steps, replayed, strict=True):
        if recorded_step.entry_hash != entry.entry_hash:
            note(f"step[{entry.idx}].entry_hash", recorded_step.entry_hash[:12],
                 entry.entry_hash[:12])
        if int(recorded_step.logical_time) != int(entry.logical_time):
            note(f"step[{entry.idx}].logical_time", recorded_step.logical_time,
                 entry.logical_time)
        if recorded_step.responder_rule != entry.responder_rule:
            note(f"step[{entry.idx}].responder_rule", recorded_step.responder_rule,
                 entry.responder_rule)
    return mismatches


def verify_manifest(
    manifest: RunManifest,
    frozen_dir: Path,
) -> VerificationReport:
    """Replay a recorded run offline and compare against its own evidence."""
    mismatches: list[Mismatch] = []
    missing: list[str] = []
    checked = 0

    for record in manifest.episodes:
        path = frozen_dir / f"{record.scenario_id}.json"
        if not path.exists():
            missing.append(record.scenario_id)
            continue
        scenario = freeze_module.load(path)
        mismatches += _compare(record, scenario)
        checked += 1

    # The aggregate block must follow from the episodes, not be asserted beside
    # them: a manifest could otherwise carry flattering metrics over honest
    # episode records.
    recomputed_metrics = aggregate(list(manifest.episodes))
    for key, value in recomputed_metrics.items():
        recorded = manifest.metrics.get(key)
        if recorded is not None and recorded != value:
            mismatches.append(
                Mismatch(
                    scenario_id="<aggregate>", field=f"metrics.{key}",
                    recorded=str(recorded), recomputed=str(value),
                ),
            )

    return VerificationReport(
        episodes_checked=checked,
        mismatches=tuple(mismatches),
        version_skew=_check_versions(manifest),
        missing_scenarios=tuple(missing),
    )
