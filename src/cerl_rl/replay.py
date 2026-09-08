"""Offline verification of a recorded pilot run. No model, no torch.

The pilot records each episode's action sequence. Replaying those actions
through the frozen scenario must reproduce every claim that is a *function of
them*: trace-head hash, terminal state hash, reward, task completion, safety,
decision correctness, violations and the verifier's tool-call count.

The first version compared **only** trace-head hashes, so a record whose reward
had been changed to 999, task completion to 1.0, safety to true and terminal
state hash to 64 ``f`` characters still returned ``ok: true`` -- and an empty
``{}`` returned ``ok: true`` with zero episodes. It verified that the actions
were real, and nothing about what was claimed for them.

Some fields genuinely cannot be reconstructed from actions, and are listed
rather than quietly accepted: gradient norms, losses, the raw token provenance
of sampling, and wall-clock timings. Those are reported as unverifiable, not
passed.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from pydantic import TypeAdapter

from cerl.actions import Action
from cerl.env.reward import W_COMMITTED_COST, W_DECISION, W_OUTCOME, W_TASK
from cerl.reference.runner import run_actions
from cerl.scenario import freeze

_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

#: Claims a replay cannot re-derive, because they are properties of the training
#: process rather than of the recorded actions.
UNVERIFIABLE_BY_REPLAY: tuple[str, ...] = (
    ("v1 only: the recorded tool_calls figure, which counted terminal "
     "declarations (see evidence/rl-pilot/metrics_audit.json for the corrected count)"),
    "gradient norms and losses (properties of the optimizer step, not the actions)",
    "advantages (derived from rewards, but recomputed here and cross-checked)",
    "the exact tokens sampled at each turn (v1 records do not retain them)",
    "wall-clock timings and memory measurements",
)

#: Every episode field this verifier reproduces from the actions.
REPRODUCED_FIELDS: tuple[str, ...] = (
    "trace_head_hash", "terminal_state_hash", "reward", "task_completion",
    "safe_completion", "decision_correct", "committed", "attempted",
)

#: v2 additionally records the verifier's own tool-call count and the action
#: total, so both are checked. v1's ``tool_calls`` counted terminal declarations
#: as tool calls (R6), so comparing it against the verifier would fail every
#: historical record for a defect that is already documented and corrected in
#: the derived audit. It is a known format difference, not a tampering signal.
V2_ONLY_FIELDS: tuple[str, ...] = ("tool_calls", "total_actions")

#: v1 records carry a resolved ``protocol`` block; v2 records carry
#: ``protocol_version`` and ``limits``. Both are real run formats and both must
#: verify -- a verifier that only understands the format it was written against
#: silently stops checking the moment the runner is upgraded.
_REQUIRED_V1 = ("protocol", "config", "baseline", "groups")
_REQUIRED_V2 = ("protocol_version", "limits", "config", "baseline", "groups")


def detect_format(record: dict[str, Any]) -> str:
    if "protocol_version" in record:
        return "v2"
    if "protocol" in record:
        return "v1"
    return "unknown"
_REQUIRED_EPISODE = ("scenario_id", "actions", "reward", "trace_head_hash")

#: Rewards are exact rationals of small integers; anything above this is a real
#: difference, not float noise.
_TOLERANCE = 1e-9


class Mismatch(dict):
    pass


def _mismatch(where: str, field: str, *, recorded: Any, replayed: Any) -> dict[str, Any]:
    return {"where": where, "field": field, "recorded": recorded, "replayed": replayed}


def replay_episode(scenario_id: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    scenario = freeze.load(freeze.FROZEN_DIR / f"{scenario_id}.json")
    parsed = tuple(_ADAPTER.validate_python(a) for a in actions)
    episode = run_actions(scenario, parsed)
    verdict = episode.verdict
    return {
        # How many of the recorded actions the environment actually executed.
        # An action appended after a terminal one is never run, so every
        # recomputed field still matches and the tampering is invisible unless
        # the count is compared too.
        "executed_actions": len(episode.trace.agent_entries()),
        "recorded_actions": len(actions),
        "trace_head_hash": episode.trace.head_hash,
        "terminal_state_hash": episode.final.state_hash(),
        "task_completion": float(verdict.task_completion),
        "safe_completion": bool(verdict.safe_completion),
        "decision_correct": bool(verdict.decision_correct),
        "committed": sorted(str(v.cost_class) for v in verdict.violations),
        "attempted": sorted(str(v.cost_class) for v in verdict.attempted_violations),
        "verifier_tool_calls": int(verdict.tool_calls),
        "reward": (
            W_OUTCOME * (1.0 if verdict.correct_final_state else 0.0)
            + W_TASK * float(verdict.task_completion)
            + W_DECISION * (1.0 if verdict.decision_correct else 0.0)
            - W_COMMITTED_COST * len(verdict.violations)
        ),
    }


def _check_episode(
    row: dict[str, Any],
    where: str,
    out: list[dict[str, Any]],
    *,
    scenario_id: str | None = None,
    run_format: str = "v1",
) -> dict[str, Any] | None:
    # Group rows carry the scenario on the group, not on each episode.
    target = row.get("scenario_id") or scenario_id
    missing = [f for f in _REQUIRED_EPISODE if f not in row and f != "scenario_id"]
    if target is None:
        missing.append("scenario_id")
    if missing:
        out.append(
            _mismatch(where, "structure", recorded=f"missing {missing}", replayed="required"),
        )
        return None
    replayed = replay_episode(target, row["actions"])
    if replayed["executed_actions"] != replayed["recorded_actions"]:
        out.append(
            _mismatch(where, "actions",
                      recorded=f"{replayed['recorded_actions']} recorded",
                      replayed=f"only {replayed['executed_actions']} executed"),
        )
    replayed["tool_calls"] = replayed["verifier_tool_calls"]
    replayed["total_actions"] = replayed["recorded_actions"]
    fields = REPRODUCED_FIELDS + (V2_ONLY_FIELDS if run_format == "v2" else ())
    for field in fields:
        if field not in row:
            continue
        recorded = row[field]
        expected = replayed[field]
        if isinstance(expected, float):
            if abs(float(recorded) - expected) > _TOLERANCE:
                out.append(_mismatch(where, field, recorded=recorded, replayed=expected))
        elif isinstance(expected, list):
            if sorted(recorded) != expected:
                out.append(_mismatch(where, field, recorded=recorded, replayed=expected))
        elif recorded != expected:
            out.append(_mismatch(where, field, recorded=recorded, replayed=expected))
    return replayed


def verify_run(path: pathlib.Path) -> dict[str, Any]:
    """Replay a recorded run and compare **everything reproducible**."""
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return {"status": "unreadable", "ok": False, "episodes_replayed": 0,
                "mismatches": [{"field": "json", "recorded": str(error)}]}
    if not isinstance(record, dict):
        return {"status": "unreadable", "ok": False, "episodes_replayed": 0,
                "mismatches": [{"field": "json", "recorded": "not an object"}]}

    mismatches: list[dict[str, Any]] = []
    run_format = detect_format(record)
    required = {"v1": _REQUIRED_V1, "v2": _REQUIRED_V2}.get(run_format, _REQUIRED_V1)
    missing = [f for f in required if f not in record]
    if missing:
        # An empty or partial record is NOT a passing verification. v1 returned
        # ok:true for "{}", which reads as "verified" and means "nothing here".
        return {
            "status": "incomplete",
            "ok": False,
            "episodes_replayed": 0,
            "mismatches": [
                _mismatch("<record>", "structure",
                          recorded=f"missing {missing}", replayed="required"),
            ],
            "note": "a record missing these keys is not a verifiable run",
        }

    checked = 0
    for phase in ("baseline", "after"):
        for row in record.get(phase, {}).get("per_episode", []):
            if _check_episode(row, phase, mismatches, run_format=run_format) is not None:
                checked += 1

    group_rewards_ok = True
    for group in record.get("groups", []):
        rewards = []
        for index, row in enumerate(group.get("episodes", [])):
            replayed = _check_episode(
                row, f"group[{group.get('update')}][{index}]", mismatches,
                scenario_id=group.get("scenario_id"), run_format=run_format,
            )
            if replayed is not None:
                checked += 1
                rewards.append(replayed["reward"])
        # The group's own reward list must agree with its episodes.
        recorded_rewards = [float(r) for r in group.get("rewards", [])]
        if rewards and len(rewards) == len(recorded_rewards):
            for a, b in zip(recorded_rewards, rewards, strict=True):
                if abs(a - b) > _TOLERANCE:
                    mismatches.append(
                        _mismatch(f"group[{group.get('update')}]", "rewards",
                                  recorded=a, replayed=b),
                    )
                    group_rewards_ok = False
        elif recorded_rewards:
            mismatches.append(
                _mismatch(f"group[{group.get('update')}]", "rewards",
                          recorded=len(recorded_rewards), replayed=len(rewards)),
            )
            group_rewards_ok = False
        # And the skip decision must follow from them.
        varied = len({round(r, 12) for r in rewards}) > 1
        if group.get("skipped") and varied:
            mismatches.append(
                _mismatch(f"group[{group.get('update')}]", "skipped",
                          recorded=True, replayed="group had reward variation"),
            )
        if not group.get("skipped") and rewards and not varied:
            mismatches.append(
                _mismatch(f"group[{group.get('update')}]", "skipped",
                          recorded=False, replayed="group had no reward variation"),
            )

    # Declared update count must match the groups.
    driven = sum(1 for g in record.get("groups", []) if not g.get("skipped"))
    if "reward_driven_updates" in record and record["reward_driven_updates"] != driven:
        mismatches.append(
            _mismatch("<record>", "reward_driven_updates",
                      recorded=record["reward_driven_updates"], replayed=driven),
        )

    # The training scenarios must be the ones the protocol selects. v1 records
    # embed the resolved selection; v2 records name the protocol version, so the
    # selection is read from that protocol instead.
    if run_format == "v2":
        from cerl_rl import protocol_v2

        declared = list(protocol_v2.training_selection().scenario_ids)
    else:
        declared = record.get("protocol", {}).get("training_scenarios", {}).get("ids")
    if declared is not None:
        used = {g["scenario_id"] for g in record.get("groups", [])}
        stray = sorted(used - set(declared))
        if stray:
            mismatches.append(_mismatch("<record>", "training_scenarios", recorded=stray,
                                        replayed="not in the declared training selection"))

    status = "verified" if not mismatches else "failed"
    if checked == 0:
        status = "empty"
    return {
        "status": status,
        "format": run_format,
        "ok": not mismatches and checked > 0,
        "episodes_replayed": checked,
        "group_rewards_consistent": group_rewards_ok,
        "mismatches": list(mismatches),
        "unverifiable_by_replay": list(UNVERIFIABLE_BY_REPLAY),
    }


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/rl-pilot/pilot_run.json")
    result = verify_run(target)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
