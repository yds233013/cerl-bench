"""A development run against a locally served open-weight model.

**What this is:** an authentic development smoke test. A real model chooses real
actions against the real environment, and the results are whatever they are.

**What this is not:** a held-out generalization experiment. It draws only from
the canonical *training* partition, because its outcomes may inform fixes, and a
run that can change the system must not consume scenarios whose value depends on
never having influenced anything. Branches training cannot supply are reported,
never borrowed from validation or evaluation.

A low score is a result to investigate. Nothing here relaxes the verifier, and
no oracle action is ever substituted for a model action -- a failed turn stays
failed and is scored as the model's behaviour.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from cerl.agents import tool_schemas
from cerl.agents.local_client import LocalModelClient
from cerl.core import Frozen, FrozenMap
from cerl.eval import latency, pilot, splits
from cerl.scenario.schema import FrozenScenario

#: The family this milestone exercises. W2 is the family with a complete oracle,
#: the fullest mutation coverage, and the approval/retry structure the research
#: study is about.
FAMILY = "duplicate_charge_approval"

#: Bounded on purpose. Measured at ~30-60 s per turn on an M2, a wider selection
#: would take hours and tell us nothing the first few episodes do not.
MAX_SCENARIOS = 5

#: Ceiling on **environment actions**. Below the environment's own 40-step
#: budget so an episode terminates within a bounded runtime; truncation is
#: scored INCOMPLETE either way.
DEFAULT_MAX_STEPS = 16

#: Ceiling on **model turns**, recorded separately because they are not the same
#: quantity: a turn that emits no tool call costs a turn and no action. Under
#: the original configuration 71% of turns fell in that gap.
DEFAULT_MAX_TURNS = 16

#: Wall-clock budget for all model calls in a run, measured outside the
#: simulator. Reaching it terminates the episode with a recorded external
#: timeout, which is an acceptable outcome -- not an error to retry away.
DEFAULT_INFERENCE_BUDGET_S = 1800.0


class BranchCoverage(Frozen):
    """What the training partition can and cannot supply for this family."""

    family: str
    selected: FrozenMap[str, int] = FrozenMap()
    unavailable: tuple[str, ...] = ()
    corpus_totals: FrozenMap[str, int] = FrozenMap()


def branch_coverage(
    scenarios: list[FrozenScenario], family: str = FAMILY,
) -> BranchCoverage:
    """Report coverage rather than repair it."""
    in_family = [s for s in scenarios if s.family == family]
    train = [s for s in in_family if splits.partition_of(s) is splits.Partition.TRAIN]

    totals: dict[str, int] = defaultdict(int)
    for scenario in in_family:
        totals[scenario.branch] += 1

    chosen = select(scenarios, family)
    selected: dict[str, int] = defaultdict(int)
    for scenario in chosen:
        selected[scenario.branch] += 1

    train_branches = {s.branch for s in train}
    return BranchCoverage(
        family=family,
        selected=FrozenMap(dict(selected)),
        unavailable=tuple(sorted(set(totals) - train_branches)),
        corpus_totals=FrozenMap(dict(totals)),
    )


def select(
    scenarios: list[FrozenScenario],
    family: str = FAMILY,
    limit: int = MAX_SCENARIOS,
) -> tuple[FrozenScenario, ...]:
    """Deterministic selection: one per available training branch, then fill.

    Sorted by scenario id, no sampling and no seed, so the set is a function of
    the corpus alone and is fixed before any result is seen. Breadth first --
    one episode from each branch training can supply -- then the remaining slots
    go to the lowest ids, so a small run still touches every reachable outcome.
    """
    pool = sorted(
        (
            s
            for s in scenarios
            if s.family == family
            and splits.partition_of(s) is splits.Partition.TRAIN
        ),
        key=lambda s: s.scenario_id,
    )
    by_branch: dict[str, list[FrozenScenario]] = defaultdict(list)
    for scenario in pool:
        by_branch[scenario.branch].append(scenario)

    chosen: list[FrozenScenario] = [by_branch[b][0] for b in sorted(by_branch)]
    for scenario in pool:
        if len(chosen) >= limit:
            break
        if scenario not in chosen:
            chosen.append(scenario)
    return tuple(sorted(chosen[:limit], key=lambda s: s.scenario_id))


class LocalRunReport(Frozen):
    """Everything the milestone must record, per episode and in aggregate."""

    provenance: str
    model: FrozenMap[str, Any]
    corpus_version: str
    split_version: str
    prompt_hash: str
    tools_hash: str
    config_hash: str
    coverage: BranchCoverage
    episodes: tuple[FrozenMap[str, Any], ...] = ()
    usage: FrozenMap[str, Any] = FrozenMap()
    #: Turn and action ceilings, recorded apart because they are different
    #: quantities and either can be the binding one.
    limits: FrozenMap[str, Any] = FrozenMap()
    #: Wall-clock inference budget and what became of it.
    deadline: FrozenMap[str, Any] = FrozenMap()
    #: How each episode ended, in words, so termination is never inferred from
    #: a missing field.
    termination: tuple[str, ...] = ()
    tool_schema_version: str = ""
    wall_seconds: float = 0.0


def config_hashes(client: LocalModelClient) -> dict[str, str]:
    """Hash the prompt, the tool schemas, and the generation settings.

    Recorded so a later run that scored differently can be told apart from a
    later run that was configured differently.
    """
    from cerl.agents.prompt_only import SYSTEM_PROMPT
    from cerl.agents.tool_schemas import tool_schema_hash
    from cerl.core import content_hash, report_hash

    info = client.info()
    return {
        "prompt_hash": content_hash(SYSTEM_PROMPT),
        "tools_hash": tool_schema_hash(),
        # ``report_hash`` rather than ``content_hash``: temperature is a float,
        # and canonical JSON forbids floats in state on purpose. A generation
        # setting is report data, and report_hash renders it at fixed precision
        # instead of smuggling a float past the state discipline.
        "config_hash": report_hash(
            {
                "model": info.model,
                "digest": info.digest,
                "num_ctx": info.num_ctx,
                "num_predict": info.num_predict,
                "temperature": info.temperature,
            },
        ),
    }


def episode_records(result: pilot.PilotResult) -> tuple[FrozenMap[str, Any], ...]:
    """Flatten a run into the per-episode facts the report must carry."""
    return tuple(
        FrozenMap(
            {
                "scenario_id": record.scenario_id,
                "family": record.family,
                "branch": record.branch,
                "partition": record.partition,
                "scenario_hash": record.scenario_hash,
                "initial_state_hash": record.initial_state_hash,
                "terminal_state_hash": record.terminal_state_hash,
                "trace_head_hash": record.trace_head_hash,
                "declared_outcome": record.declared_outcome,
                "required_decision": record.required_decision,
                "failure_class": str(record.failure_class),
                "task_completion": record.task_completion,
                "correct_final_state": record.correct_final_state,
                "decision_correct": record.decision_correct,
                "safe_task_completion": record.safe_completion,
                # Kept apart at every level, here included.
                "committed_violations": len(record.committed_violations),
                "attempted_violations": len(record.attempted_violations),
                "prohibited_side_effects": len(record.prohibited_side_effects),
                "tool_calls": record.tool_calls,
                "oracle_tool_calls": record.oracle_tool_calls,
                "truncated": record.truncated,
                "steps": len(record.actions),
            },
        )
        for record in result.manifest.episodes
    )


def run(
    scenarios: list[FrozenScenario],
    client: LocalModelClient,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    limit: int = MAX_SCENARIOS,
) -> tuple[pilot.PilotResult, LocalRunReport]:
    """Run the selection sequentially and build the report.

    Termination is recorded, never inferred: an episode ends by declaring an
    outcome, by exhausting its action limit, or by the inference deadline. All
    three are acceptable results for this milestone.
    """
    chosen = list(select(scenarios, FAMILY, limit))
    hashes = config_hashes(client)
    hashes["tool_schema_version"] = tool_schemas.TOOL_SCHEMA_VERSION
    with latency.measure() as elapsed:
        result = pilot.execute(
            chosen, client, None, source="local", max_steps=max_steps,
        )

    info = client.info()
    report = LocalRunReport(
        provenance=info.provenance,
        model=FrozenMap(info.model_dump(mode="json")),
        corpus_version=chosen[0].corpus_version if chosen else "",
        split_version=splits.SPLIT_VERSION_CANONICAL,
        coverage=branch_coverage(scenarios, FAMILY),
        episodes=episode_records(result),
        usage=FrozenMap(client.usage()),
        limits=FrozenMap(
            {
                "max_environment_actions": max_steps,
                "max_model_turns": DEFAULT_MAX_TURNS,
                "env_budget_steps": chosen[0].budget_steps if chosen else None,
            },
        ),
        deadline=FrozenMap(_deadline_report(client)),
        termination=tuple(_termination(result, client)),
        wall_seconds=round(elapsed.seconds, 2),
        **hashes,
    )
    return result, report


def _deadline_report(client: LocalModelClient) -> dict[str, Any]:
    """Empty when no budget was set, rather than fabricating one."""
    deadline = getattr(client, "deadline", None)
    return dict(deadline.report()) if deadline is not None else {}


def _termination(result: pilot.PilotResult, client: LocalModelClient) -> list[str]:
    """One line per episode saying how it ended."""
    lines: list[str] = []
    for record in result.manifest.episodes:
        if record.declared_outcome:
            lines.append(
                f"{record.scenario_id}: declared {record.declared_outcome}",
            )
        elif record.truncated:
            lines.append(f"{record.scenario_id}: environment step budget exhausted")
        else:
            lines.append(f"{record.scenario_id}: action limit reached without a decision")
    for entry in result.interrupted:
        if "DeadlineExceeded" in entry or "timed out" in entry.lower():
            note = _deadline_report(client)
            lines.append(
                f"EXTERNAL TIMEOUT at the inference deadline | {entry} | "
                f"spent {note.get('spent_seconds')}s of "
                f"{note.get('budget_seconds')}s | "
                f"{note.get('cancellation_note', '')}",
            )
        else:
            lines.append(f"{entry} | interrupted")
    return lines


def write_report(report: LocalRunReport, path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
