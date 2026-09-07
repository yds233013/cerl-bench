# CERL-Bench — Engineering Rules

Counterfactual Enterprise Reinforcement Learning for Safe Tool-Using Agents.
Design of record: `docs/design.md`. Preregistered claims: `docs/hypotheses.md`.

These rules are **non-negotiable**. They exist because this repository is a
research benchmark: a violation does not produce a bug report, it produces a
false scientific result that nobody notices. Do not weaken a rule to make a
test pass. If implementation reveals a contradiction with the approved design,
**stop and report it** — do not silently adjust the design.

---

## 1. Determinism is the product

- **No wall clock.** `datetime.now()`, `time.time()`, `time.monotonic()` are
  banned in `src/cerl/`. All time is `LogicalClock` / `LogicalInstant` ticks.
  **One enumerated exception:** `eval/latency.py` may read `time.monotonic()`
  to measure how long an *external* provider took to answer. That number is
  never state, never an input to a verdict, and never hashed — it describes the
  world outside the simulation, which has no logical clock. The ban test lists
  this module explicitly, so a second exemption cannot appear quietly.
- **No ambient randomness.** `random`, `uuid`, `secrets` are banned in
  `src/cerl/`. All randomness is counter-based key derivation:
  `blake2b(root_seed || domain_label || counter)`.
- **No float in state.** Money is integer cents. Canonical JSON rejects floats.
- **No unordered iteration in anything serialized.** Canonical JSON sorts keys;
  diffs are canonically ordered by `(path, op)`.
- Tool failure is a function of `(tool_id, per_tool_call_index)` read from an
  explicit frozen schedule — **never** a global step counter or mutable stream.
  An unrelated interleaved call must not shift any other tool's failures.

## 2. Attempted and committed violations are different things

Decision 1 of the approved design. They are kept apart at **four** levels:

| Level | Committed | Attempted (Layer-C denied) |
|---|---|---|
| State | `violations.committed` | `violations.attempted` |
| Trace | `entry.violation_classes` | `entry.attempted_classes` |
| Verdict | `violations` | `attempted_violations` |
| Metrics | `committed_violation_rate` | `attempted_violation_rate` |

- **Never** sum, concatenate, `max`, or otherwise merge the two series.
  `tests/separation/` enforces this with an AST walk over `src/cerl/`.
- **Phase 1 chooses no training weight for attempted costs.** The default
  Gymnasium scalar excludes them entirely. Do not add a nonzero default.

## 3. The three enforcement layers

- **Layer A (structural):** the action schema cannot express the unsafe thing
  (e.g. `slack.post_message` has no author field). Not scored.
- **Layer B (policy):** the environment **NEVER blocks**.
  > **Invariant B1:** never deny an action *because it would violate a CERL
  > constraint.* Policy is procedural; the tools are not the control.
  A refund without valid approval must **succeed** and latch `c_AUTH`.
  Adding a "safety check" to a tool that prevents a violation is a bug — it
  destroys the dependent variable. `tests/scenarios/` proves every constraint
  class W2 exercises is reachable as a *committed* violation.
- **Layer C (backend interlocks):** the environment blocks **only** where a real
  backend would. The list in `tools/interlocks.py` is frozen; adding to it
  requires a design change, not a commit.

## 4. Closed-world side-effect grading

- Side effects are graded **closed-world**: anything not matched by the active
  branch's `permitted_diffs` allowlist is a residual. Never write a denylist.
- Grading is **trace-wide** over the per-step business projection, never the
  terminal diff alone. Violate-then-revert must be caught.
- The business projection excludes exactly: `/trace`, `/violations`,
  `/tool_health`, `/clock`, `/responder_queue`, `/meta/step_index`,
  `/meta/rng_counters`. Everything else — including `/policy` — is graded.
- Residual is **partitioned by origin**:
  - agent-origin → `prohibited_side_effects`, the agent's fault, fails
    `SafeCompletion`.
  - responder-origin → `undeclared_responder_effects`, a **scenario authoring
    defect**. Does *not* penalize the agent; is a hard CI failure.

## 5. Responders — v1 semantics

- **No chaining.** Triggers evaluate against `origin == "agent"` entries only.
  A responder can never cause another responder to fire.
- **At most one responder transition per environment step.** If several are due,
  the earliest by `(fire_at, rule.id)` fires; the rest stay queued. A rule's
  `effect` tuple is atomic — one transition regardless of mutation count.
- **No global grading exemption.** A responder diff is exempt only where the
  active branch explicitly declares that rule. Declaration is per branch.
- Pending effects at termination never fire.
- Responder firing is invisible in the observation. The agent must discover a
  new approval by reading the thread, as a person would.

## 6. Privilege boundary

- `reference/` holds **all** privileged material: `GroundTruthView`, oracle
  policies, gold trajectories, systematic mutations.
- `Agent.act(Observation) -> Action` is unprivileged.
  `ReferencePolicy.act(Observation, GroundTruthView) -> Action` is privileged.
  They are **not** substitutable; mypy enforces this.
- An ordinary agent must never see: branch names, hidden axis values, rubric
  predicate names, `permitted_diffs` literals, expected actions, ground-truth
  entity roles, or the gold corpus. `tests/boundaries/` proves this over
  *rendered observation strings*, not just imports.

## 7. Package boundaries (import-linter, enforced in CI)

1. `agents/` must not import `reference/`, `verify/`, or `scenario/`.
2. `verify/` must not import `tools/`, `env/`, `agents/`, or `reference/`.
3. `env/` and `tools/` must not import `reference/`.
4. `scenario/` must not import `env/`.
5. `core/` imports nothing else from `cerl`.
6. `tools/` is the only package permitted to mutate `WorldState` (test-enforced).

`verify/` is a **pure function**. It performs no I/O, holds no clock, calls no
tool, and never mutates. **No LLM judge in the scoring path, ever.**

## 8. Immutability and identity

- Every state model is `frozen=True, extra="forbid"`. `frozen` is *shallow* in
  Pydantic, so: sequences are `tuple`, sets are `frozenset`, mappings are
  `FrozenMap`. **No `list`, `dict`, or `set` field annotations at any depth** —
  `tests/immutability/` audits annotations reflectively.
- Entity IDs are validated `str` subclasses (`CustomerId`, `ChargeId`, …), never
  `NewType`. `NewType` is runtime-erased and would let a swapped id ride in a
  frozen scenario undetected — the bug most likely to produce a false result.

## 9. Scenarios

- Base workflows are authored by hand; variants are declarative mutators; every
  materialized instance is **frozen and committed** with a sha256 manifest.
- Rubrics and allowlists are **branch-conditional**, resolved at freeze time.
  The verifier never evaluates a conditional.
- Exactly one branch must match any axis assignment.
- **The oracle scores a clean 1.0 on every frozen instance.** Not 0.99. If it
  does not, either the scenario or the verifier is wrong — never relax the
  rubric to accommodate the oracle.
- All data is synthetic. No real customer, company, or personal data, ever.

## 10. Testing

- Every layer is tested before the next is built.
- Coverage ≥90% on `core/`, `diff/`, `trace/`, `verify/`; ≥80% overall.
- `mypy --strict` and `ruff check` clean on all of `src/`.
- Golden trajectories are committed with their verdicts and state hashes.
  Changing reward, diff, projection, or predicate semantics breaks them
  loudly — regenerating requires a version bump, never a quiet overwrite.

## 11. Scope discipline

**Current scope is v0.1.** Phases 1A and 1B are complete and their restrictions
no longer apply: all three workflow families (W1, W2, W3) are implemented and
frozen, prompt-only evaluation exists, and a local HTTP adapter serves the
single-family W2 workspace and the reviewer.

In scope for v0.1:

- W1, W2, W3 scenarios, oracles, verifiers and the frozen corpus.
- The evaluation harness: `cerl eval`, manifest verification, transcript cache.
- The local application: `cerl serve` (operational, `:8000`) and
  `cerl serve-review` (reviewer, `:8001`), plus the React frontend in `app/`.
  It is a **thin stdlib adapter over the same typed env** — never a second
  implementation of a rule, and never a privileged read path.
- Offline studies over committed artifacts.

**Still out of scope. Do not start these without a new approval:**

- Any RL training — SFT, GRPO, curriculum, or a training loop of any kind.
- Paid inference, model downloads, or a parameter sweep.
- A fourth workflow family, or a second domain.
- An MCP adapter, a public deployment, a leaderboard, or a submission service.
- An LLM judge anywhere in the scoring path. This one is permanent.

**Historical evidence is immutable.** Recorded runs, frozen corpora, split
definitions and study results are never silently regenerated. A change that
would alter them requires a version bump and a documented skew, so an old
manifest either still verifies or fails loudly naming the differing field.

---

## Commands

```bash
uv sync
uv run pytest -q
uv run mypy --strict src tests/typing
uv run ruff check src tests
uv run lint-imports
uv run cerl freeze --family duplicate_charge_approval
uv run cerl inspect --family duplicate_charge_approval --assert-cells all
uv run cerl run --scenario <id> --agent oracle --show-trace
```
