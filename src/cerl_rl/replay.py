"""Offline replay of the pilot's recorded episodes. No model, no torch.

The pilot records the action sequence of every episode. Replaying those actions
through the frozen scenario must reproduce the trace head hash and the terminal
state hash exactly. That is the same guarantee the benchmark makes for its own
runs, applied to this experiment: if a reported reward cannot be re-derived from
the recorded actions, it is not evidence of anything.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from pydantic import TypeAdapter

from cerl.actions import Action
from cerl.reference.runner import run_actions
from cerl.scenario import freeze

_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def replay_episode(scenario_id: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    scenario = freeze.load(freeze.FROZEN_DIR / f"{scenario_id}.json")
    parsed = tuple(_ADAPTER.validate_python(a) for a in actions)
    episode = run_actions(scenario, parsed)
    return {
        "trace_head_hash": episode.trace.head_hash,
        "terminal_state_hash": episode.final.state_hash(),
        "safe_completion": bool(episode.verdict.safe_completion),
        "task_completion": float(episode.verdict.task_completion),
    }


def verify_run(path: pathlib.Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    checked = 0
    mismatches: list[dict[str, Any]] = []

    def check(scenario_id: str, row: dict[str, Any], where: str) -> None:
        nonlocal checked
        if "actions" not in row:
            return
        replayed = replay_episode(scenario_id, row["actions"])
        checked += 1
        if replayed["trace_head_hash"] != row.get("trace_head_hash"):
            mismatches.append(
                {"where": where, "scenario_id": scenario_id, "field": "trace_head_hash",
                 "recorded": row.get("trace_head_hash"), "replayed": replayed["trace_head_hash"]},
            )

    for phase in ("baseline", "after"):
        for row in record.get(phase, {}).get("per_episode", []):
            check(row["scenario_id"], row, phase)
    for group in record.get("groups", []):
        for row in group.get("episodes", []):
            check(group["scenario_id"], row, f"group[{group['update']}]")

    return {"episodes_replayed": checked, "mismatches": mismatches, "ok": not mismatches}


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "evidence/rl-pilot/pilot_run.json")
    result = verify_run(target)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
