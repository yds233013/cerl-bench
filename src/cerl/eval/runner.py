"""Run policies over a partition and record everything a verifier-free reader needs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cerl.actions import Action
from cerl.core import FrozenMap
from cerl.diff import Origin
from cerl.env import CerlEnv
from cerl.eval.manifest import AgentConfig, EpisodeRecord, RecordedStep, RunManifest
from cerl.eval.metrics import aggregate
from cerl.eval.splits import SPLIT_VERSION, partition_of
from cerl.reference.registry import alternative_for, oracle_for
from cerl.reference.runner import Episode, run_actions, run_reference
from cerl.scenario.freeze import scenario_hash
from cerl.scenario.schema import FrozenScenario
from cerl.verify.verifier import predicate_library_hash


def git_sha(repo: Path | None = None) -> str:
    """Best-effort commit id for the manifest; empty if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _record_steps(episode: Episode) -> tuple[RecordedStep, ...]:
    return tuple(
        RecordedStep(
            idx=entry.idx,
            origin=entry.origin,
            actor=str(entry.actor),
            logical_time=entry.logical_time,
            action_kind=entry.action_kind,
            action=entry.action if entry.origin is Origin.AGENT else None,
            responder_rule=entry.responder_rule,
            outcome=str(entry.outcome),
            denied_interlock=entry.denied_interlock,
            committed_classes=tuple(str(c) for c in entry.violation_classes),
            attempted_classes=tuple(str(c) for c in entry.attempted_classes),
            state_hash_before=entry.state_hash_before,
            state_hash_after=entry.state_hash_after,
            entry_hash=entry.entry_hash,
        )
        for entry in episode.trace.entries
    )


def record_episode(scenario: FrozenScenario, episode: Episode) -> EpisodeRecord:
    verdict = episode.verdict
    return EpisodeRecord(
        scenario_id=scenario.scenario_id,
        template_id=scenario.template_id,
        family=scenario.family,
        branch=scenario.branch,
        partition=partition_of(scenario).value,
        scenario_hash=scenario_hash(scenario),
        initial_state_hash=scenario.world.state_hash(),
        actions=episode.actions,
        steps=_record_steps(episode),
        terminal_state_hash=episode.final.state_hash(),
        trace_head_hash=episode.trace.head_hash,
        declared_outcome=verdict.declared_outcome,
        required_decision=verdict.required_decision,
        failure_class=verdict.failure_class,
        rubric=verdict.rubric,
        correct_final_state=verdict.correct_final_state,
        task_completion=verdict.task_completion,
        decision_correct=verdict.decision_correct,
        safe_completion=verdict.safe_completion,
        committed_violations=verdict.violations,
        attempted_violations=verdict.attempted_violations,
        prohibited_side_effects=tuple(o.path for o in verdict.prohibited_side_effects),
        undeclared_responder_effects=tuple(
            o.path for o in verdict.undeclared_responder_effects
        ),
        tool_calls=verdict.tool_calls,
        oracle_tool_calls=verdict.oracle_tool_calls,
        truncated=episode.truncated,
        responder_transitions=len(episode.trace.responder_entries()),
    )


REFERENCE_POLICIES = {"oracle": oracle_for, "alternative": alternative_for}


def run_reference_evaluation(
    scenarios: list[FrozenScenario],
    policy_name: str = "oracle",
) -> RunManifest:
    """Evaluate a reference policy. Privileged; never a benchmark arm."""
    if policy_name not in REFERENCE_POLICIES:
        raise KeyError(f"unknown reference policy {policy_name!r}")
    factory = REFERENCE_POLICIES[policy_name]
    records = [
        record_episode(scenario, run_reference(scenario, factory(scenario)))
        for scenario in scenarios
    ]
    return _manifest(
        records,
        AgentConfig(
            name=policy_name,
            kind="reference",
            # Recorded honestly: a reference policy reads ground truth, so its
            # numbers are an upper bound and not a benchmark result.
            privilege_mode="privileged",
        ),
        scenarios,
    )


def run_recorded_actions(
    scenarios: list[FrozenScenario],
    actions_by_scenario: dict[str, tuple[Action, ...]],
    agent: AgentConfig,
) -> RunManifest:
    """Re-run recorded actions. No model, no network."""
    records = []
    for scenario in scenarios:
        actions = actions_by_scenario.get(scenario.scenario_id)
        if actions is None:
            continue
        records.append(record_episode(scenario, run_actions(scenario, actions)))
    return _manifest(records, agent, scenarios)


def _manifest(
    records: list[EpisodeRecord],
    agent: AgentConfig,
    scenarios: list[FrozenScenario],
) -> RunManifest:
    return RunManifest(
        git_sha=git_sha(),
        predicate_library_hash=predicate_library_hash(),
        split_version=SPLIT_VERSION,
        agent=agent,
        families=tuple(sorted({s.family for s in scenarios})),
        seeds=tuple(sorted({s.root_seed for s in scenarios})),
        episodes=tuple(records),
        metrics=FrozenMap(aggregate(records)),
    )


def env_for(scenario: FrozenScenario) -> CerlEnv:
    return CerlEnv(scenario)
