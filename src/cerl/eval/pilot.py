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

**Split integrity.** The pilot draws only from **training-eligible** scenarios:
the ``train`` partition *minus* every scenario carrying a value registered as
held out in ``siblings``. Partition membership alone was not enough -- 85 of the
144 train scenarios are registered counterfactuals, because a CF and its ID
sibling deliberately share a partition. See ``splits.ELIGIBILITY_VERSION`` and
``docs/pilot-split-audit.md``.

It is a *development smoke test*: its outcomes may inform fixes, so it must not
touch anything whose value depends on never having informed anything. The
consequence is stated rather than engineered around -- **eligible scenarios reach
8 of the 10 outcome branches, not 10.** Two W3 branches are reachable only
through held-out values, and importing one to restore coverage is exactly the
move that would make the pilot leak.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cerl.agents import tool_schemas
from cerl.agents.budget import (
    HeuristicTokenCounter,
    SpendJournal,
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
    request_key,
)
from cerl.agents.prompt_only import (
    SYSTEM_PROMPT,
    AgentTranscript,
    PromptOnlyAgent,
    transcript_entries,
)
from cerl.agents.tool_schemas import all_tool_schemas
from cerl.core import Frozen, FrozenMap, content_hash
from cerl.env.env import CerlEnv
from cerl.env.render import render_observation
from cerl.eval import splits
from cerl.eval.agent_harness import run_agent
from cerl.eval.manifest import AgentConfig, RunManifest
from cerl.eval.runner import record_episode
from cerl.reference.ground_truth import ground_truth_for
from cerl.reference.registry import oracle_for
from cerl.scenario import siblings
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

#: Restrict to training-eligible scenarios. Off only for diagnostics: with it
#: off, the selection can include registered counterfactuals.
PILOT_ELIGIBLE_ONLY = True

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
    eligible_only: bool = True
    #: Selected scenarios carrying a registered held-out value. Must be empty.
    held_out_selected: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """Leakage-free. Deliberately does **not** require full branch coverage.

        Missing branches are a corpus limitation to report, not a defect in the
        selection -- and treating them as one would create pressure to fix the
        "failure" by importing a held-out value.
        """
        return not self.held_out_partitions_touched and not self.held_out_selected


def select(
    scenarios: list[FrozenScenario],
    partition: splits.Partition | None = PILOT_PARTITION,
    per_branch: int = EPISODES_PER_BRANCH,
    *,
    eligible_only: bool = PILOT_ELIGIBLE_ONLY,
) -> tuple[FrozenScenario, ...]:
    """Pick ``per_branch`` scenarios from every outcome branch of the pool.

    Deterministic: the lowest scenario ids in sort order, no sampling and no
    seed, so the set is a function of the corpus and cannot drift between the
    proposal and the run.

    Branches the eligible pool cannot supply are simply absent. They are
    reported by :func:`audit`, never backfilled from a held-out value.
    """
    pool = splits.select(scenarios, partition)
    if eligible_only:
        pool = splits.training_eligible(pool)
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
    *,
    eligible_only: bool = PILOT_ELIGIBLE_ONLY,
) -> SelectionAudit:
    """Check the selection against the split manifest.

    Reports rather than repairs. A branch the partition cannot supply is
    surfaced as ``uncoverable_branches`` -- the conflict is stated, not resolved
    by quietly reaching into a held-out partition.
    """
    chosen = select(scenarios, partition, per_branch, eligible_only=eligible_only)
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
        eligible_only=eligible_only,
        held_out_selected=tuple(
            sorted(
                s.scenario_id
                for s in chosen
                if siblings.is_held_out(s.axes, s.template_id)
            ),
        ),
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
    *,
    eligible_only: bool = PILOT_ELIGIBLE_ONLY,
) -> PilotProjection:
    """Select, audit, and estimate. Offline and free."""
    counter = counter or HeuristicTokenCounter()
    chosen = select(scenarios, partition, per_branch, eligible_only=eligible_only)
    return PilotProjection(
        episodes=tuple(profile(s, counter) for s in chosen),
        audit=audit(scenarios, partition, per_branch, eligible_only=eligible_only),
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
    #: Requests recovered with no outcome. Charged, never replayed.
    orphaned_requests: tuple[str, ...] = ()
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
    ledger: SpendLedger | None = None,
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

    An episode that raises -- budget exhausted, an ambiguous request outcome,
    a local model running out of context -- stops the run and is recorded in
    ``interrupted``. Episodes already completed are kept and scored: a partial
    run is a real result, and discarding it would waste what was already spent
    or computed.

    ``ledger`` is optional because a locally served model costs nothing to run.
    Metering something that has no price would produce a spend report that reads
    as authoritative and means nothing.
    """
    records = []
    transcripts: list[AgentTranscript] = []
    interrupted: list[str] = []

    for scenario in scenarios:
        agent = PromptOnlyAgent(client, scenario_id=scenario.scenario_id)
        env = CerlEnv(scenario)
        run = run_agent(env, agent, max_steps=max_steps)
        transcripts.append(run.transcript or agent.transcript)
        # Score whatever was executed, interrupted or not. The actions really
        # happened, so the episode is recorded and its termination reason kept;
        # dropping it would erase evidence the environment already committed.
        records.append(record_episode(scenario, _episode_for(scenario, run.actions)))
        if run.failure is not None:
            interrupted.append(f"{scenario.scenario_id}: {run.failure}")
            break
        if ledger is not None and ledger_path is not None:
            ledger.save(ledger_path)

    manifest = _pilot_manifest(records, scenarios, client, ledger, source, max_steps)
    if ledger is not None and ledger_path is not None:
        ledger.save(ledger_path)
    return PilotResult(
        manifest=manifest,
        transcripts=tuple(transcripts),
        spend=FrozenMap(ledger.report() if ledger is not None else {}),
        completed=len(records),
        interrupted=tuple(interrupted),
        orphaned_requests=tuple(ledger.orphaned_requests) if ledger is not None else (),
        source=source,
    )


def _provider_label(client: ModelClient, source: str) -> str:
    """Who produced these actions, from the object that produced them.

    Never a constant chosen at the call site: a synthetic run labelled
    ``anthropic``, or a local run labelled ``live``, is the one mistake this
    field exists to prevent.
    """
    if source == "live":
        return str(getattr(client, "name", "unknown"))
    if source == "local":
        return str(getattr(client, "provenance", getattr(client, "name", "local")))
    return "synthetic-transport"


def _episode_for(scenario: FrozenScenario, actions: tuple[Any, ...]) -> Any:
    from cerl.reference.runner import run_actions

    return run_actions(scenario, actions)


def _pilot_manifest(
    records: list[Any],
    scenarios: list[FrozenScenario],
    client: ModelClient,
    ledger: SpendLedger | None,
    source: str,
    max_steps: int,
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
            provider=_provider_label(client, source),
            model=getattr(client, "model", PILOT_MODEL),
            # The cap actually used, not the module default. Regeneration
            # replays with this number, and a wrong one walks off the end of
            # the transcript and reports a cache miss that is really a bug here.
            max_steps=max_steps,
            max_retries=PILOT_MAX_RETRIES,
            tool_schema_version=tool_schemas.TOOL_SCHEMA_VERSION,
            tools_hash=tool_schemas.tool_schema_hash(),
            prompt_hash=content_hash(SYSTEM_PROMPT),
        ),
        scenarios,
    )
    return manifest.model_copy(
        update={
            # A synthetic run is never a Claim 2 artifact. Recording the source
            # in the manifest means a reader cannot mistake one for the other.
            "verification_mode": source,
            "metrics": FrozenMap(
                {
                    **dict(manifest.metrics),
                    **({"spend": ledger.report()} if ledger is not None else {}),
                },
            ),
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
    #: Set when the run was recorded under a different agent-visible schema.
    schema_mismatch: str = ""
    #: Set when the run ended outside the environment -- a deadline, a kill --
    #: so a short transcript is expected rather than evidence of a bad cache.
    terminated_early: str = ""

    @property
    def ok(self) -> bool:
        return not self.mismatched and not self.cache_misses and not self.schema_mismatch

    def summary(self) -> str:
        lines = [f"regenerated {self.matched}/{self.scenarios} episodes from cache"]
        if self.schema_mismatch:
            lines.append(f"  SCHEMA SKEW {self.schema_mismatch}")
        if self.terminated_early:
            lines.append(f"  ENDED EARLY {self.terminated_early}")
        lines.extend(f"  MISMATCH {entry}" for entry in self.mismatched)
        lines.extend(f"  CACHE MISS {entry}" for entry in self.cache_misses)
        return "\n".join(lines)


def _first_key_mismatch(
    by_id: dict[str, FrozenScenario], transcripts: list[AgentTranscript],
) -> str:
    """Whether a transcript's first request key still reproduces here."""
    for transcript in transcripts:
        scenario = by_id.get(transcript.scenario_id)
        if scenario is None or not transcript.entries:
            continue
        env = CerlEnv(scenario)
        first = [
            {"role": "user", "content": render_observation(env.reset())},
        ]
        expected = request_key(SYSTEM_PROMPT, first, all_tool_schemas())
        if expected != transcript.entries[0].request_key:
            return (
                f"{transcript.scenario_id}: the recorded first request key "
                f"{transcript.entries[0].request_key[:16]} does not reproduce "
                f"here (now {expected[:16]}). This run predates tool-schema "
                f"versioning and was recorded against a different prompt or "
                f"schema; current tool schema is "
                f"{tool_schemas.TOOL_SCHEMA_VERSION}. Version skew, not a "
                f"broken cache."
            )
    return ""


def _schema_skew(recorded_version: str, recorded_hash: str, current_hash: str) -> str:
    version = recorded_version or "(unversioned)"
    return (
        f"run recorded tool schema {version} hash {recorded_hash[:16]}; this "
        f"checkout has {tool_schemas.TOOL_SCHEMA_VERSION} hash "
        f"{current_hash[:16]}. Request keys include the schemas, so these "
        f"transcripts cannot be regenerated here. This is version skew, not a "
        f"broken cache."
    )


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

    # A schema change moves every request key, so every entry would "miss".
    # Reporting that as a cache miss blames the cache for a version skew.
    recorded_hash = manifest.agent.tools_hash
    current_hash = tool_schemas.tool_schema_hash()
    if recorded_hash and recorded_hash != current_hash:
        return RegenerationReport(
            scenarios=len(transcripts),
            matched=0,
            schema_mismatch=_schema_skew(
                manifest.agent.tool_schema_version, recorded_hash, current_hash,
            ),
        )

    # Runs recorded before the schema was versioned carry no hash to compare, so
    # detect skew structurally: recompute the first request key and see whether
    # it matches what was recorded. If it does not, the transcript was made
    # against a different prompt or tool schema, and every entry will "miss".
    # Reporting that as a cache miss blames the cache for version skew.
    if not recorded_hash and transcripts:
        stale = _first_key_mismatch(by_id, transcripts)
        if stale:
            return RegenerationReport(
                scenarios=len(transcripts), matched=0, schema_mismatch=stale,
            )
    # Replay under the run's own step cap. An episode that never declared an
    # outcome ends by exhausting its budget, so a different cap changes where
    # it stops -- and replaying past the end looks like a cache miss when it is
    # really a configuration mismatch.
    max_steps = manifest.agent.max_steps or PILOT_MAX_STEPS
    mismatched: list[str] = []
    misses: list[str] = []
    matched = 0

    early: list[str] = []
    for transcript in transcripts:
        scenario = by_id.get(transcript.scenario_id)
        if scenario is None:
            misses.append(f"{transcript.scenario_id}: no such frozen scenario")
            continue
        client = TranscriptCacheClient(transcript_entries(transcript))
        agent = PromptOnlyAgent(client, scenario_id=scenario.scenario_id)
        env = CerlEnv(scenario)
        # Replay no further than the transcript actually goes. A run cut short
        # from outside -- an inference deadline, a kill -- recorded fewer turns
        # than its step cap, and walking past the end would report a cache miss
        # for a cache that is complete up to where the run stopped. A genuine
        # miss *within* the transcript still surfaces.
        recorded_turns = len(transcript.entries)
        completed = recorded.get(transcript.scenario_id)
        # Bound replay by the transcript only when the episode did *not*
        # complete. If the manifest records a finished episode, its transcript
        # must cover it, and a short one is a genuine cache defect rather than
        # an interruption.
        bound = max_steps if completed is not None else min(max_steps, recorded_turns)
        try:
            run = run_agent(env, agent, max_steps=bound)
        except TranscriptCacheMiss as miss:
            misses.append(f"{transcript.scenario_id}: {miss}")
            continue

        expected = completed
        if expected is None:
            # No manifest episode: the run ended before this episode finished.
            # The turns it did record replayed, which is what can be verified.
            early.append(
                f"{transcript.scenario_id}: no completed episode in the "
                f"manifest; {recorded_turns} recorded model turns replayed to "
                f"the point the run stopped",
            )
            matched += 1
            continue
        if recorded_turns < max_steps and not run.actions:
            early.append(f"{transcript.scenario_id}: transcript ends at turn {recorded_turns}")
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
        terminated_early="; ".join(early),
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


def journal_path_for(ledger_path: Path) -> Path:
    """The write-ahead log that sits beside a ledger snapshot.

    Two files, two jobs: the snapshot carries cap and model provenance to check
    a resume against, the journal carries the money at per-request granularity.
    """
    return ledger_path.with_suffix(".jsonl")


def open_ledger(
    ledger_path: Path, cap_cents: float, model: str = PILOT_MODEL,
) -> SpendLedger:
    """Resume or start a ledger with durable per-request recording enabled."""
    journal = journal_path_for(ledger_path)
    if ledger_path.exists() or journal.exists():
        return SpendLedger.resume(ledger_path, cap_cents, model, journal_path=journal)
    return SpendLedger(
        cap_cents=cap_cents, model=model, journal=SpendJournal(journal),
    )
