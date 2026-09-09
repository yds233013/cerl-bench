"""Corrected metrics for a recorded run, derived rather than edited in place.

The runner counted every non-malformed action as a ``tool_call``, terminal
declarations included, which disagrees with the benchmark verifier's definition:
15 reported against 10 for a validation phase, 191 against 150 in training.

The original rows keep their recorded values -- rewriting evidence to match a
later understanding is how a record stops being one. This module recomputes the
metrics from the recorded actions and writes a **separate** audit, so the two
can be compared and the discrepancy is itself part of the record.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from pydantic import TypeAdapter

from cerl.actions import Action
from cerl.reference.runner import run_actions
from cerl.scenario import freeze
from cerl_rl.environment import TERMINAL_KINDS, ActionCategory

_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def categorise_recorded(actions: list[dict[str, Any]]) -> dict[str, int]:
    """Split recorded actions into the three categories the verifier implies."""
    counts = {c.value: 0 for c in ActionCategory}
    for action in actions:
        kind = str(action.get("kind"))
        if kind == "malformed":
            counts[ActionCategory.MALFORMED.value] += 1
        elif kind in TERMINAL_KINDS:
            counts[ActionCategory.TERMINAL.value] += 1
        else:
            counts[ActionCategory.TOOL_CALL.value] += 1
    return counts


def _phase(rows: list[dict[str, Any]], scenario_id: str | None = None) -> dict[str, Any]:
    reported = 0
    verifier_tool_calls = 0
    totals = {c.value: 0 for c in ActionCategory}
    for row in rows:
        reported += int(row.get("tool_calls", 0))
        target = row.get("scenario_id") or scenario_id
        scenario = freeze.load(freeze.FROZEN_DIR / f"{target}.json")
        actions = tuple(_ADAPTER.validate_python(a) for a in row["actions"])
        verifier_tool_calls += int(run_actions(scenario, actions).verdict.tool_calls)
        for key, value in categorise_recorded(row["actions"]).items():
            totals[key] += value
    return {
        "reported_tool_calls": reported,
        "verifier_tool_calls": verifier_tool_calls,
        "tool_calls": totals[ActionCategory.TOOL_CALL.value],
        "terminal_declarations": totals[ActionCategory.TERMINAL.value],
        "malformed_actions": totals[ActionCategory.MALFORMED.value],
        "total_actions": sum(totals.values()),
    }


def audit(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "what_this_is": (
            "Metrics recomputed from the recorded actions. The original record is "
            "unchanged; where 'reported_tool_calls' differs from "
            "'verifier_tool_calls', the runner counted terminal declarations as "
            "tool calls and the verifier does not."
        ),
        "phases": {},
    }
    for phase in ("baseline", "after"):
        rows = record.get(phase, {}).get("per_episode", [])
        if rows:
            out["phases"][phase] = _phase(rows)
    training: list[dict[str, Any]] = [
        {**row, "scenario_id": group["scenario_id"]}
        for group in record.get("groups", [])
        for row in group.get("episodes", [])
    ]
    if training:
        out["phases"]["training"] = _phase(training)
    return out


def main() -> int:
    import sys

    source = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else "evidence/rl-pilot/pilot_run.json",
    )
    result = audit(json.loads(source.read_text()))
    destination = source.parent / "metrics_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
