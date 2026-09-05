"""The live-evaluation pilot: selection, projection, execution, and replay.

Four capabilities, deliberately separable so the expensive one is the only one
that needs authorisation:

* **Selection** -- which scenarios the pilot covers, fixed by a rule and audited
  against the split manifest. Offline.
* **Projection** -- what the run would cost, estimated from the real request
  payloads. Offline, and an *estimate*: no live call has ever been made, so no
  figure here is a measured price.
* **Execution** -- the run itself. Gated on explicit authorisation and a
  configured budget, metered per request, sequential.
* **Regeneration** -- replaying a recorded run from its transcript cache,
  offline and byte-exact.

**Split integrity.** The pilot draws only from the ``train`` partition. It is a
*development smoke test*: its outcomes may inform fixes, so it must not touch
scenarios whose value depends on never having informed anything. The other
partitions are not "development data" and are not renamed -- they are simply not
used here. See ``docs/pilot-split-audit.md``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cerl.agents.budget import (
    HeuristicTokenCounter,
    SpendLedger,
    TokenCounter,
    cost_cents,
)
from cerl.agents.model_client import (
    AnthropicClient,
    ModelClient,
    TranscriptCacheClient,
    TranscriptCacheMiss,
    Transport,
)
from cerl.agents.prompt_only import (
    SYSTEM_PROMPT,
    AgentTranscript,
    PromptOnlyAgent,
    transcript_entries,
)
from cerl.agents.tool_schemas import all_tool_schemas
from cerl.core import Frozen, FrozenMap
from cerl.env.env import CerlEnv
from cerl.env.render import render_observation
from cerl.eval import splits
from cerl.eval.agent_harness import run_agent
from cerl.eval.manifest import AgentConfig, RunManifest
from cerl.eval.runner import record_episode
from cerl.reference.ground_truth import ground_truth_for
from cerl.reference.registry import oracle_for
from cerl.scenario.schema import FrozenScenario

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

#: Episodes per outcome branch. Two distinguishes "handles this branch" from
#: "one episode went well"; it supports no per-branch significance claim.
EPISODES_PER_BRANCH = 2

#: The partition the pilot draws from. Not a default to be overridden casually:
#: a pilot whose outcomes inform fixes must not consume held-out scenarios.
PILOT_PARTITION = splits.Partition.TRAIN

PILOT_MODEL = "claude-opus-5"
PILOT_MAX_TOKENS = 2048
PILOT_MAX_STEPS = 40
PILOT_MAX_RETRIES = 2

#: Assumed assistant output per turn in the *expected* projection. The worst
#: case uses the full ``max_tokens``.
EXPECTED_OUTPUT_TOKENS = 500
#: Thinking allowance folded into each turn's transcript growth.
THINKING_ALLOWANCE = 200


class PilotNotAuthorized(RuntimeError):
    """Execution was attempted without authorisation and a configured budget."""


# --------------------------------------------------------------------------
# selection, audited against the split manifest
# --------------------------------------------------------------------------


class SelectionAudit(Frozen):
    """What the selection touches, so integrity is checkable rather than claimed."""

    partition: str
    scenario_ids: tuple[str, ...]
    partitions_touched: FrozenMap[str, int]
    branch_coverage: FrozenMap[str, int]
    sibling_groups: tuple[str, ...]
    #: Branches the chosen partition cannot supply. Empty is the passing state.
    uncoverable_branches: tuple[str, ...]
    held_out_partitions_touched: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.held_out_partitions_touched and not self.uncoverable_branches


def select(
    scenarios: list[FrozenScenario],
    partition: splits.Partition | None = PILOT_PARTITION,
    per_branch: int = EPISODES_PER_BRANCH,
) -> tuple[FrozenScenario, ...]:
    """Pick ``per_branch`` scenarios from every outcome branch of one partition.

    Deterministic: the lowest scenario ids in sort order, no sampling and no
    seed, so the set is a function of the corpus and cannot drift between the
    proposal and the run.
    """
    pool = splits.select(scenarios, partition)
    by_branch: dict[tuple[str, str], list[FrozenScenario]] = defaultdict(list)
    for scenario in pool:
        by_branch[(scenario.family, scenario.branch)].append(scenario)
    chosen: list[FrozenScenario] = []
    for key in sorted(by_branch):
        chosen.extend(sorted(by_branch[key], key=lambda s: s.scenario_id)[:per_branch])
    return tuple(chosen)


def audit(
    scenarios: list[FrozenScenario],
    partition: splits.Partition | None = PILOT_PARTITION,
    per_branch: int = EPISODES_PER_BRANCH,
) -> SelectionAudit:
    """Check the selection against the split manifest.

    Reports rather than repairs. A branch the partition cannot supply is
    surfaced as ``uncoverable_branches`` -- the conflict is stated, not resolved
    by quietly reaching into a held-out partition.
    """
    chosen = select(scenarios, partition, per_branch)
    all_branches = {f"{s.family}/{s.branch}" for s in scenarios}
    covered = {f"{s.family}/{s.branch}" for s in chosen}

    touched: dict[str, int] = defaultdict(int)
    for scenario in chosen:
        touched[splits.partition_of(scenario).value] += 1

    coverage: dict[str, int] = defaultdict(int)
    for scenario in chosen:
        coverage[f"{scenario.family}/{scenario.branch}"] += 1

    wanted = partition.value if partition is not None else None
    held_out = tuple(sorted(p for p in touched if wanted is not None and p != wanted))

    return SelectionAudit(
        partition=wanted or "all",
        scenario_ids=tuple(s.scenario_id for s in chosen),
        partitions_touched=FrozenMap(dict(touched)),
        branch_coverage=FrozenMap(dict(coverage)),
        sibling_groups=tuple(sorted({splits.pair_key(s) for s in chosen})),
        uncoverable_branches=tuple(sorted(all_branches - covered)),
        held_out_partitions_touched=held_out,
    )


# --------------------------------------------------------------------------
# projection -- an estimate, never a measured price
# --------------------------------------------------------------------------


class EpisodeProjection(Frozen):
    """Estimated cost profile for one pilot episode.

    Derived from serialised payloads driven by the oracle. **Estimated, not
    measured**: no live call has been made, and a real model's trajectory will
    differ in both length and token shape.
    """

    scenario_id: str
    family: str
    branch: str
    partition: str
    oracle_steps: int
    budget_steps: int
    mean_turn_tokens: int
    max_turn_tokens: int

    def estimated_cents(
        self, *, worst_case: bool, cached: bool, max_tokens: int = PILOT_MAX_TOKENS,
    ) -> float:
        steps = self.budget_steps if worst_case else self.oracle_steps
        per_turn = self.max_turn_tokens if worst_case else self.mean_turn_tokens
        output = max_tokens if worst_case else EXPECTED_OUTPUT_TOKENS
        total = 0.0
        for turn in range(1, steps + 1):
            prefix = fixed_prefix_tokens() + (turn - 1) * per_turn
            if cached:
                total += cost_cents(
                    PILOT_MODEL,
                    0,
                    output,
                    cache_read_tokens=prefix,
                    cache_write_tokens=per_turn,
                )
            else:
                total += cost_cents(PILOT_MODEL, prefix + per_turn, output)
        return total


class PilotProjection(Frozen):
    episodes: tuple[EpisodeProjection, ...]
    audit: SelectionAudit
    model: str = PILOT_MODEL
    max_tokens: int = PILOT_MAX_TOKENS
    #: The counter the projection used. A heuristic estimate carries wider
    #: uncertainty than a provider count, and the report says which it was.
    counter_name: str = HeuristicTokenCounter.name

    @property
    def branches(self) -> tuple[str, ...]:
        return tuple(sorted(self.audit.branch_coverage))

    def total_cents(self, *, worst_case: bool, cached: bool = True) -> float:
        return sum(
            e.estimated_cents(
                worst_case=worst_case, cached=cached, max_tokens=self.max_tokens,
            )
            for e in self.episodes
        )

    def recommended_cap_cents(self) -> int:
        """A cap the *modelled* worst case cannot exceed, rounded up to a dollar.

        "Modelled" is the caveat that matters: the worst case assumes every
        episode runs to its step budget with every response at ``max_tokens``.
        That bounds the request count and the per-request output, which is what
        the ledger enforces. It does not bound input tokenization error --
        ``docs/budget-accounting.md`` §Residual exposure.
        """
        return int((self.total_cents(worst_case=True, cached=True) // 100 + 1) * 100)


def fixed_prefix_tokens(counter: TokenCounter | None = None) -> int:
    """System prompt plus tool schemas: resent (or cached) on every request."""
    counter = counter or HeuristicTokenCounter()
    return counter.count(SYSTEM_PROMPT, [], list(all_tool_schemas()))


def profile(scenario: FrozenScenario, counter: TokenCounter | None = None) -> EpisodeProjection:
    """Estimate one episode's prompt growth by running the oracle offline."""
    counter = counter or HeuristicTokenCounter()
    env = CerlEnv(scenario)
    observation = env.reset()
    policy = oracle_for(scenario)
    truth = ground_truth_for(scenario)
    turns: list[int] = []
    while not env.done:
        user_turn = {"role": "user", "content": render_observation(observation)}
        rendered = counter.count("", [user_turn], [])
        action = policy.act(observation, truth)
        reply_turn = {"role": "assistant", "content": str(action.model_dump(mode="json"))}
        reply = counter.count("", [reply_turn], [])
        turns.append(rendered + reply + THINKING_ALLOWANCE)
        observation = env.step(action).observation
    return EpisodeProjection(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        branch=scenario.branch,
        partition=splits.partition_of(scenario).value,
        oracle_steps=len(turns),
        budget_steps=scenario.budget_steps,
        mean_turn_tokens=sum(turns) // len(turns),
        max_turn_tokens=max(turns),
    )


def project(
    scenarios: list[FrozenScenario],
    partition: splits.Partition | None = PILOT_PARTITION,
    per_branch: int = EPISODES_PER_BRANCH,
    counter: TokenCounter | None = None,
) -> PilotProjection:
    """Select, audit, and estimate. Offline and free."""
    counter = counter or HeuristicTokenCounter()
    chosen = select(scenarios, partition, per_branch)
    return PilotProjection(
        episodes=tuple(profile(s, counter) for s in chosen),
        audit=audit(scenarios, partition, per_branch),
        counter_name=counter.name,
    )


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


class PilotResult(Frozen):
    """What a pilot run produced, including what it cost and what is unknown."""

    manifest: RunManifest
    transcripts: tuple[AgentTranscript, ...]
    spend: FrozenMap[str, Any]
    completed: int
    interrupted: tuple[str, ...] = ()
    #: "live" or "synthetic", from the transport. Never from a flag.
    source: str = "synthetic"


def build_client(
    ledger: SpendLedger,
    transport: Transport | None = None,
    counter: TokenCounter | None = None,
    model: str = PILOT_MODEL,
    max_tokens: int = PILOT_MAX_TOKENS,
) -> AnthropicClient:
    """Construct the metered client.

    Authorisation is required only for a live transport. A synthetic transport
    spends nothing, so gating it would only make the offline path untestable --
    and an untested spending path is a worse risk than a testable free one.
    """
    synthetic = transport is not None and getattr(transport, "source", "live") != "live"
    return AnthropicClient(
        model=model,
        max_tokens=max_tokens,
        ledger=ledger,
        transport=transport,
        counter=counter,
        max_retries=PILOT_MAX_RETRIES,
        require_authorization=not synthetic,
    )


def execute(
    scenarios: list[FrozenScenario],
    client: ModelClient,
    ledger: SpendLedger,
    *,
    source: str = "synthetic",
    max_steps: int = PILOT_MAX_STEPS,
    ledger_path: Path | None = None,
) -> PilotResult:
    """Run the pilot **sequentially**, one request outstanding at a time.

    Sequential by choice, not by limitation: with a single request in flight the
    spendable balance at each decision point is unambiguous and the ledger reads
    as an auditable line-by-line account. Concurrency would buy wall-clock time
    at the cost of the one property that makes the spending claim checkable.

    An episode that raises -- budget exhausted, an ambiguous request outcome --
    stops the run and is recorded in ``interrupted``. Episodes already completed
    are kept and scored: a partial pilot is a real result, and discarding it
    would waste money already spent.
    """
    records = []
    transcripts: list[AgentTranscript] = []
    interrupted: list[str] = []

    for scenario in scenarios:
        agent = PromptOnlyAgent(client, scenario_id=scenario.scenario_id)
        env = CerlEnv(scenario)
        try:
            run = run_agent(env, agent, max_steps=max_steps)
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            interrupted.append(f"{scenario.scenario_id}: {type(error).__name__}: {error}")
            transcripts.append(agent.transcript)
            break
        transcripts.append(run.transcript or agent.transcript)
        records.append(record_episode(scenario, _episode_for(scenario, run.actions)))
        if ledger_path is not None:
            ledger.save(ledger_path)

    manifest = _pilot_manifest(records, scenarios, client, ledger, source)
    if ledger_path is not None:
        ledger.save(ledger_path)
    return PilotResult(
        manifest=manifest,
        transcripts=tuple(transcripts),
        spend=FrozenMap(ledger.report()),
        completed=len(records),
        interrupted=tuple(interrupted),
        source=source,
    )


def _episode_for(scenario: FrozenScenario, actions: tuple[Any, ...]) -> Any:
    from cerl.reference.runner import run_actions

    return run_actions(scenario, actions)


def _pilot_manifest(
    records: list[Any],
    scenarios: list[FrozenScenario],
    client: ModelClient,
    ledger: SpendLedger,
    source: str,
) -> RunManifest:
    from cerl.eval.runner import _manifest

    manifest = _manifest(
        records,
        AgentConfig(
            name="prompt_only",
            kind="model",
            privilege_mode="unprivileged",
            # The transport, not the client class. A synthetic run recorded as
            # "anthropic" would read as a provider call that never happened.
            provider=(
                getattr(client, "name", "unknown") if source == "live" else "synthetic-transport"
            ),
            model=getattr(client, "model", PILOT_MODEL),
            max_steps=PILOT_MAX_STEPS,
            max_retries=PILOT_MAX_RETRIES,
        ),
        scenarios,
    )
    return manifest.model_copy(
        update={
            # A synthetic run is never a Claim 2 artifact. Recording the source
            # in the manifest means a reader cannot mistake one for the other.
            "verification_mode": "live" if source == "live" else "synthetic",
            "metrics": FrozenMap({**dict(manifest.metrics), "spend": ledger.report()}),
        },
    )


# --------------------------------------------------------------------------
# regeneration from the transcript cache
# --------------------------------------------------------------------------


class RegenerationReport(Frozen):
    """Whether a recorded run replays byte-exactly from its own transcripts."""

    scenarios: int
    matched: int
    mismatched: tuple[str, ...] = ()
    cache_misses: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatched and not self.cache_misses

    def summary(self) -> str:
        lines = [f"regenerated {self.matched}/{self.scenarios} episodes from cache"]
        lines.extend(f"  MISMATCH {entry}" for entry in self.mismatched)
        lines.extend(f"  CACHE MISS {entry}" for entry in self.cache_misses)
        return "\n".join(lines)


def regenerate(
    scenarios: list[FrozenScenario],
    manifest: RunManifest,
    transcripts: list[AgentTranscript],
) -> RegenerationReport:
    """Replay a recorded run from its transcripts, offline and byte-exact.

    This is the *model* reproducibility claim, and it is deliberately weaker and
    separate from the environment's replay guarantee. Replaying actions proves
    the environment and verifier are deterministic; replaying transcripts proves
    only that the same recorded turns produce the same actions. A cache miss is
    a hard failure rather than a fall-through to a live call -- a cache that
    quietly reaches the network is not a reproducibility mechanism.
    """
    by_id = {s.scenario_id: s for s in scenarios}
    recorded = {e.scenario_id: e.actions for e in manifest.episodes}
    mismatched: list[str] = []
    misses: list[str] = []
    matched = 0

    for transcript in transcripts:
        scenario = by_id.get(transcript.scenario_id)
        if scenario is None:
            misses.append(f"{transcript.scenario_id}: no such frozen scenario")
            continue
        client = TranscriptCacheClient(transcript_entries(transcript))
        agent = PromptOnlyAgent(client, scenario_id=scenario.scenario_id)
        env = CerlEnv(scenario)
        try:
            run = run_agent(env, agent, max_steps=PILOT_MAX_STEPS)
        except TranscriptCacheMiss as miss:
            misses.append(f"{transcript.scenario_id}: {miss}")
            continue
        expected = recorded.get(transcript.scenario_id)
        if expected is None:
            misses.append(f"{transcript.scenario_id}: not in the manifest")
            continue
        if tuple(run.actions) != tuple(expected):
            mismatched.append(
                f"{transcript.scenario_id}: {len(run.actions)} actions replayed, "
                f"{len(expected)} recorded",
            )
            continue
        matched += 1

    return RegenerationReport(
        scenarios=len(transcripts),
        matched=matched,
        mismatched=tuple(mismatched),
        cache_misses=tuple(misses),
    )


def write_transcripts(transcripts: tuple[AgentTranscript, ...], path: Path) -> Path:
    """Commit the transcript cache beside the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transcripts": [t.model_dump(mode="json") for t in transcripts],
        "sources": sorted({t.source for t in transcripts}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_transcripts(path: Path) -> tuple[AgentTranscript, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        AgentTranscript.model_validate(entry) for entry in payload["transcripts"]
    )
