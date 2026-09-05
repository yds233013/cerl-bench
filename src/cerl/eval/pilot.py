"""Live-evaluation pilot: deterministic selection and measured cost projection.

Two things live here, both offline and free:

* **Selection.** Which scenarios the pilot covers, fixed by a rule rather than by
  sampling, so the set is reproducible from the corpus alone and is settled
  before any result is seen.
* **Projection.** What the run would cost, measured by serialising the exact
  request payloads the harness builds and driving each episode with the oracle.
  An estimate derived from the real prompts is worth having; a guess is not.

Executing the pilot is deliberately *not* here. See ``docs/live-pilot-proposal.md``.
"""

from __future__ import annotations

from collections import defaultdict

from cerl.agents.budget import cost_cents, estimate_input_tokens
from cerl.agents.prompt_only import SYSTEM_PROMPT
from cerl.agents.tool_schemas import all_tool_schemas
from cerl.core import Frozen, canonical_json
from cerl.env.env import CerlEnv
from cerl.env.render import render_observation
from cerl.reference.ground_truth import ground_truth_for
from cerl.reference.registry import oracle_for
from cerl.scenario.schema import FrozenScenario

#: Episodes per outcome branch. Two distinguishes "handles this branch" from
#: "one episode went well"; it supports no per-branch significance claim.
EPISODES_PER_BRANCH = 2

#: Proposed configuration (config C in the proposal).
PILOT_MODEL = "claude-opus-5"
PILOT_MAX_TOKENS = 2048
#: A turn's assistant output, for the expected case. The worst case uses the
#: full ``max_tokens``.
EXPECTED_OUTPUT_TOKENS = 500
#: Thinking-token allowance folded into each turn's transcript growth.
THINKING_ALLOWANCE = 200

#: US$/million tokens. Cache read is 0.1x input, cache write 1.25x.
CACHE_READ_PER_MTOK = 0.5
CACHE_WRITE_PER_MTOK = 6.25


class EpisodeProjection(Frozen):
    """Measured cost profile for one pilot episode."""

    scenario_id: str
    family: str
    branch: str
    oracle_steps: int
    budget_steps: int
    #: Transcript growth per turn, measured from the rendered payloads.
    mean_turn_tokens: int
    max_turn_tokens: int

    def cost_cents(self, *, worst_case: bool, cached: bool) -> float:
        """Cost of this episode, in cents.

        The expected case runs the oracle-length trajectory at typical output
        size. The worst case drives the episode to its full step budget with
        every response hitting ``max_tokens`` -- which is what a spending cap
        must actually be set from, since input grows with the transcript and a
        flailing episode costs far more than an efficient one.
        """
        steps = self.budget_steps if worst_case else self.oracle_steps
        per_turn = self.max_turn_tokens if worst_case else self.mean_turn_tokens
        output = PILOT_MAX_TOKENS if worst_case else EXPECTED_OUTPUT_TOKENS
        total = 0.0
        for turn in range(1, steps + 1):
            prefix = _fixed_prefix_tokens() + (turn - 1) * per_turn
            if cached:
                total += prefix * CACHE_READ_PER_MTOK / 1_000_000
                total += per_turn * CACHE_WRITE_PER_MTOK / 1_000_000
                total += cost_cents(PILOT_MODEL, 0, output) / 100
            else:
                total += cost_cents(PILOT_MODEL, prefix + per_turn, output) / 100
        return total * 100


class PilotProjection(Frozen):
    """The whole pilot: what it covers and what it would cost."""

    episodes: tuple[EpisodeProjection, ...]
    model: str = PILOT_MODEL
    max_tokens: int = PILOT_MAX_TOKENS

    @property
    def branches(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({(e.family, e.branch) for e in self.episodes}))

    def total_cents(self, *, worst_case: bool, cached: bool = True) -> float:
        return sum(e.cost_cents(worst_case=worst_case, cached=cached) for e in self.episodes)

    def recommended_cap_cents(self) -> int:
        """A cap the worst case cannot exceed, rounded up to a whole dollar."""
        worst = self.total_cents(worst_case=True, cached=True)
        return int((worst // 100 + 1) * 100)


def _fixed_prefix_tokens() -> int:
    """The system prompt plus tool schemas, resent (or cached) every request."""
    blob = canonical_json(
        {"system": SYSTEM_PROMPT, "messages": [], "tools": list(all_tool_schemas())},
    )
    return estimate_input_tokens(len(blob))


def select(scenarios: list[FrozenScenario]) -> tuple[FrozenScenario, ...]:
    """Pick the pilot set: ``EPISODES_PER_BRANCH`` from every outcome branch.

    Deterministic by construction -- the lowest scenario ids in sort order, no
    sampling and no seed -- so the selection is reproducible from the corpus and
    cannot drift between the proposal and the run.
    """
    by_branch: dict[tuple[str, str], list[FrozenScenario]] = defaultdict(list)
    for scenario in scenarios:
        by_branch[(scenario.family, scenario.branch)].append(scenario)
    chosen: list[FrozenScenario] = []
    for key in sorted(by_branch):
        ranked = sorted(by_branch[key], key=lambda s: s.scenario_id)
        chosen.extend(ranked[:EPISODES_PER_BRANCH])
    return tuple(chosen)


def profile(scenario: FrozenScenario) -> EpisodeProjection:
    """Measure one episode's prompt growth by running the oracle offline."""
    env = CerlEnv(scenario)
    observation = env.reset()
    policy = oracle_for(scenario)
    truth = ground_truth_for(scenario)
    turns: list[int] = []
    while not env.done:
        rendered = estimate_input_tokens(len(render_observation(observation)))
        action = policy.act(observation, truth)
        reply = estimate_input_tokens(
            len(canonical_json(action.model_dump(mode="json"))),
        )
        turns.append(rendered + reply + THINKING_ALLOWANCE)
        observation = env.step(action).observation
    return EpisodeProjection(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        branch=scenario.branch,
        oracle_steps=len(turns),
        budget_steps=scenario.budget_steps,
        mean_turn_tokens=sum(turns) // len(turns),
        max_turn_tokens=max(turns),
    )


def project(scenarios: list[FrozenScenario]) -> PilotProjection:
    """Select the pilot set and measure what it would cost."""
    return PilotProjection(episodes=tuple(profile(s) for s in select(scenarios)))
