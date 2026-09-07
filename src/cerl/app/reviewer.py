"""The reviewer API: recorded episodes, with verdicts. A separate process.

This serves exactly what the operational app must not: verdicts, violation
series, branch labels, failure classes. That is why it is its own server and its
own command rather than a route behind a flag. Hiding a panel would leave the
data on the wire; a process that is not running cannot serve anything.

It reads **recorded artifacts only** and never steps an environment forward with
a model in the loop, so nothing here can consume inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cerl.app.http import ApiError, Router
from cerl.eval import manifest as manifest_module
from cerl.scenario.schema import FrozenScenario

#: Where recorded runs live, and what kind of thing each one is. The kind is
#: carried through to the UI: a scripted demonstration, a hand-authored
#: adversarial case and an actual model run are three different kinds of
#: evidence and must not be shown as one.
SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "local-baseline",
        "evidence/local-baseline/local_run.json",
        "model-run",
    ),
    (
        "local-v3-recovered",
        "evidence/local-v3/RECOVERED_partial_episode.json",
        "model-run-partial",
    ),
)

KIND_LABELS: dict[str, str] = {
    "model-run": "Actual model run",
    "model-run-partial": "Actual model run (interrupted -- partial observation)",
    "scripted-demo": "Scripted demonstration",
    "adversarial": "Constructed adversarial case",
}


class ReviewLibrary:
    """Recorded episodes, loaded from disk."""

    def __init__(self, scenarios: list[FrozenScenario], root: Path = Path()) -> None:
        self._scenarios = {s.scenario_id: s for s in scenarios}
        self._root = root
        self._episodes: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for run_id, relative, kind in SOURCES:
            path = self._root / relative
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "recovered_episode" in payload:
                episodes = [payload["recovered_episode"]]
                provenance = payload.get("note", "")
                termination = payload.get("termination", "")
                agent: dict[str, Any] = {}
            else:
                manifest = manifest_module.load(path)
                episodes = [
                    json.loads(e.model_dump_json()) for e in manifest.episodes
                ]
                agent = json.loads(manifest.agent.model_dump_json())
                provenance = (
                    f"{agent.get('provider', 'unknown')} · "
                    f"schema {agent.get('tool_schema_version') or '1.0.0 (unversioned)'}"
                )
                termination = ""
            for index, episode in enumerate(episodes):
                key = f"{run_id}:{index}"
                self._episodes[key] = {
                    "episode_id": key,
                    "run_id": run_id,
                    "kind": kind,
                    "kind_label": KIND_LABELS[kind],
                    "provenance": provenance,
                    "termination": termination,
                    "agent": agent,
                    "episode": episode,
                }

    def router(self) -> Router:
        router = Router()
        router.add("GET", "/review/health", self._health)
        router.add("GET", "/review/episodes", self._list)
        router.add("GET", "/review/episodes/{episode_id}", self._detail)
        return router

    def _health(self, _p: dict[str, str], _b: dict[str, Any]) -> tuple[int, Any]:
        return 200, {
            "status": "ok",
            "mode": "reviewer",
            "episodes": len(self._episodes),
        }

    def _list(self, _p: dict[str, str], _b: dict[str, Any]) -> tuple[int, Any]:
        rows = []
        for key, entry in sorted(self._episodes.items()):
            episode = entry["episode"]
            rows.append(
                {
                    "episode_id": key,
                    "run_id": entry["run_id"],
                    "kind": entry["kind"],
                    "kind_label": entry["kind_label"],
                    "scenario_id": episode.get("scenario_id"),
                    "branch": episode.get("branch"),
                    "failure_class": episode.get("failure_class"),
                    # Task and safety, never merged into one badge.
                    "task_completion": episode.get("task_completion"),
                    "safe_task_completion": episode.get("safe_completion"),
                    "committed_violations": len(episode.get("committed_violations") or []),
                    "attempted_violations": len(episode.get("attempted_violations") or []),
                    "steps": len(episode.get("actions") or []),
                }
            )
        return 200, {"episodes": rows}

    def _detail(self, params: dict[str, str], _b: dict[str, Any]) -> tuple[int, Any]:
        entry = self._episodes.get(params["episode_id"])
        if entry is None:
            raise ApiError(404, "no such recorded episode")
        episode = entry["episode"]
        steps = episode.get("steps") or []
        actions = episode.get("actions") or []
        timeline = []
        for index, step in enumerate(steps):
            action = step.get("action") or (
                actions[index] if index < len(actions) else {}
            )
            timeline.append(
                {
                    "index": step.get("idx", index),
                    "origin": step.get("origin"),
                    "actor": step.get("actor"),
                    "logical_time": step.get("logical_time"),
                    "kind": step.get("action_kind"),
                    "arguments": {
                        k: v for k, v in (action or {}).items() if k != "kind"
                    },
                    "outcome": step.get("outcome"),
                    "denied_interlock": step.get("denied_interlock"),
                    # The two series stay apart at every level, here included.
                    "committed_classes": step.get("committed_classes") or [],
                    "attempted_classes": step.get("attempted_classes") or [],
                    "state_hash_before": step.get("state_hash_before"),
                    "state_hash_after": step.get("state_hash_after"),
                    "responder_rule": step.get("responder_rule"),
                }
            )
        return 200, {
            "episode_id": entry["episode_id"],
            "run_id": entry["run_id"],
            "kind": entry["kind"],
            "kind_label": entry["kind_label"],
            "provenance": entry["provenance"],
            "termination": entry["termination"] or _termination_of(episode),
            "agent": entry["agent"],
            "scenario_id": episode.get("scenario_id"),
            "branch": episode.get("branch"),
            "required_decision": episode.get("required_decision"),
            "declared_outcome": episode.get("declared_outcome"),
            "task": {
                "task_completion": episode.get("task_completion"),
                "correct_final_state": episode.get("correct_final_state"),
                "decision_correct": episode.get("decision_correct"),
                "rubric": episode.get("rubric") or {},
            },
            "safety": {
                "safe_task_completion": episode.get("safe_completion"),
                "failure_class": episode.get("failure_class"),
                "committed_violations": episode.get("committed_violations") or [],
                "attempted_violations": episode.get("attempted_violations") or [],
                "prohibited_side_effects": episode.get("prohibited_side_effects") or [],
            },
            "hashes": {
                "scenario_hash": episode.get("scenario_hash"),
                "initial_state_hash": episode.get("initial_state_hash"),
                "terminal_state_hash": episode.get("terminal_state_hash"),
                "trace_head_hash": episode.get("trace_head_hash"),
            },
            "timeline": timeline,
        }


def _termination_of(episode: dict[str, Any]) -> str:
    if episode.get("declared_outcome"):
        return f"declared {episode['declared_outcome']}"
    if episode.get("truncated"):
        return "environment step budget exhausted"
    return "action limit reached without a declaration"
