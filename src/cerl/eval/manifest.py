"""The run manifest: what a result record must contain to be verifiable.

The manifest exists so a number can be checked rather than believed. It records
the scenario's identity *and its initial state's identity*, the agent
configuration and privilege mode, every action and tool result, logical times,
responder transitions, state hashes, verdicts, both violation series separately,
and the aggregate metrics.

Two reproducibility claims live here and are deliberately kept apart:

* **Offline replay** -- given the recorded actions, replaying them through the
  frozen scenario reproduces every state hash, every responder transition and
  the verdict, with no model and no network. This is a property of the
  environment and verifier, and it is guaranteed.
* **Transcript-cache regeneration** -- replaying a model's recorded
  request/response pairs reproduces the *actions*. This is a property of the
  agent and its provider, and it is a separate capability with its own failure
  modes.

Conflating them would let a green replay imply a model result was reproduced
when it was not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cerl.actions import Action
from cerl.core import Frozen, FrozenMap, LogicalInstant, report_hash
from cerl.diff import Origin
from cerl.env.render import RENDERER_VERSION
from cerl.scenario.generator import GENERATOR_VERSION
from cerl.state import AttemptedViolation, LatchedViolation
from cerl.verify import VERIFIER_VERSION
from cerl.verify.taxonomy import FailureClass
from cerl.verify.verifier import predicate_library_hash

MANIFEST_VERSION = "1.0.0"


class RecordedStep(Frozen):
    """One trace entry, in the form a verifier-free reader can check."""

    idx: int
    origin: Origin
    actor: str
    logical_time: LogicalInstant
    action_kind: str
    action: Action | None = None
    responder_rule: str | None = None
    outcome: str
    denied_interlock: str | None = None
    committed_classes: tuple[str, ...] = ()
    attempted_classes: tuple[str, ...] = ()
    state_hash_before: str
    state_hash_after: str
    entry_hash: str


class EpisodeRecord(Frozen):
    scenario_id: str
    template_id: str
    family: str
    branch: str
    partition: str

    # Scenario identity AND initial-state identity: a scenario file could be
    # edited without changing its id, so the world it produced is hashed too.
    scenario_hash: str
    initial_state_hash: str

    actions: tuple[Action, ...]
    steps: tuple[RecordedStep, ...]
    terminal_state_hash: str
    trace_head_hash: str

    declared_outcome: str | None
    required_decision: str
    failure_class: FailureClass
    rubric: FrozenMap[str, bool]
    correct_final_state: bool
    task_completion: float
    decision_correct: bool
    safe_completion: bool

    # Two independent series, never merged.
    committed_violations: tuple[LatchedViolation, ...] = ()
    attempted_violations: tuple[AttemptedViolation, ...] = ()

    prohibited_side_effects: tuple[str, ...] = ()
    undeclared_responder_effects: tuple[str, ...] = ()
    tool_calls: int
    oracle_tool_calls: int | None
    truncated: bool
    responder_transitions: int


class AgentConfig(Frozen):
    """What produced the actions, recorded so a run can be reproduced or refuted."""

    name: str
    kind: str  # "reference" | "model"
    privilege_mode: str  # "privileged" | "unprivileged"
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_steps: int | None = None
    #: Ceiling on model turns, which is not the same as environment actions: a
    #: turn that emits no tool call costs a turn and no action.
    max_turns: int | None = None
    max_retries: int | None = None
    transcript_cache: str | None = None
    #: The agent-visible schema this run was shown. A transcript recorded under
    #: a different version cannot be regenerated, because the request key
    #: includes the schemas.
    tool_schema_version: str = ""
    tools_hash: str = ""
    prompt_hash: str = ""


class RunManifest(Frozen):
    manifest_version: str = MANIFEST_VERSION
    git_sha: str = ""
    generator_version: str = GENERATOR_VERSION
    verifier_version: str = VERIFIER_VERSION
    predicate_library_hash: str = ""
    renderer_version: str = RENDERER_VERSION
    split_version: str = ""

    agent: AgentConfig
    partition: str | None = None
    families: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()

    #: Which reproducibility claim this run supports. "replay" is guaranteed;
    #: "transcript_cache" is a separate, weaker claim; "live" is not guaranteed.
    verification_mode: str = "replay"

    episodes: tuple[EpisodeRecord, ...] = ()
    metrics: FrozenMap[str, Any] = FrozenMap()

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(e.scenario_id for e in self.episodes)


def write(manifest: RunManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = report_hash(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load(path: Path) -> RunManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("manifest_hash", None)
    return RunManifest.model_validate(payload)


def current_versions() -> dict[str, str]:
    return {
        "generator_version": GENERATOR_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "predicate_library_hash": predicate_library_hash(),
        "renderer_version": RENDERER_VERSION,
    }
