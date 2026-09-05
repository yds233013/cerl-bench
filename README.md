# CERL-Bench

**Counterfactual Enterprise Reinforcement Learning for Safe Tool-Using Agents**

A research benchmark and RL environment for testing whether agents that succeed
at enterprise tool-use learn *transferable safe decision procedures*, or merely
fit the state distributions they were trained on.

Answering that needs an environment where a single controlled perturbation — an
approval that expired, a profile whose external reference does not match, a
refund request from an address that is not the account contact — flips the
correct behaviour while leaving surface complexity unchanged. Everything here
follows from that requirement.

## Status: Phase 1B

Three workflow families, complete end to end, with a deterministic verifier,
hash-chained traces, exact offline replay, and an evaluation harness.

| Family | Scenarios | Branches |
|---|---|---|
| `duplicate_charge_approval` (W2) | 114 | refund_now (16), request_then_refund (13), refund_below_threshold (49), escalate_unapproved (36) |
| `duplicate_billing_profile` (W1) | 36 | merge_sanctioned (8), distinct_entities (8), escalate_ambiguous (20) |
| `suspicious_refund_escalation` (W3) | 40 | legitimate_refund (10), request_info (10), escalate_fraud (20) |

**Canonical status: [`docs/status.md`](docs/status.md).** In short — Phase 1B is
implemented and verified with two recorded scope deviations: W1's
`identity_evidence` axis is a separately reported challenge set rather than a
matched pair (Criterion 41 passes on 104 pairs, which exclude it), and live model
evaluation is `BLOCKED_EXTERNAL` with a pilot proposed but not executed.

**Not implemented**: SFT, GRPO, curriculum arms, HTTP/MCP adapters, frontend.
No C1–C6 research claim has been measured.

**Control baselines** are measured in [`docs/baselines.md`](docs/baselines.md).
The headline: no always-escalate policy solves an entire family, but escalation
is not free either — a policy that merely declares escalation safely completes
0/190, and the best escalating control reaches 36/190, solving exactly one branch
of one family.

## Quick start

```bash
uv sync
uv run pytest -q

# Freeze every family and verify the oracle scores 1.0 on all of them
uv run cerl freeze

# Coverage and partitions
uv run cerl inspect --assert-cells all
uv run cerl splits

# Run one scenario under a reference policy
uv run cerl run --scenario <id> --agent oracle --show-trace
uv run cerl run --scenario <id> --agent alternative

# Evaluate a partition and verify the result offline, with no model involved
uv run cerl eval --partition evaluation --agent oracle --out runs/oracle.json
uv run cerl verify-manifest runs/oracle.json

# Watch the five core behaviours
uv run cerl demo
```

Per-family slices:

```bash
uv run cerl run --agent oracle --show-trace --scenario \
  dup_charge_threshold__amt-above_threshold__appr-missing_obtainable__ttl-standard__nd-present_similar_name__pp-none__t-10000__tr-stable__s17

uv run cerl run --agent oracle --show-trace --scenario \
  dup_profile_reconciliation__disp-none__id-strong_match__appr-valid__pp-none__tr-stable__s17

uv run cerl run --agent oracle --show-trace --scenario \
  suspicious_refund__bait-present__conf-none__sig-2__kind-identity__tr-stable__s17
```

## What makes it a benchmark rather than a simulator

- **The oracle scores a clean 1.0 on every frozen scenario.** Not 0.99. That
  simultaneously proves every scenario is solvable, that the branch-conditional
  rubric resolves coherently, and that the closed-world allowlist is not too
  tight for the intended solution.
- **Two materially different correct trajectories per scenario**, under a stated
  rule that rejects padding: neither may be a subsequence of the other, and they
  must differ in tool selection, in the order of mutating actions, or in the set
  of state changes made.
- **The environment never blocks on policy grounds.** A refund without valid
  approval *succeeds* and latches a violation. Blocking it would prevent
  violations rather than measure them.
- **Closed-world side-effect grading, trace-wide**, partitioned by origin, so a
  revert cannot launder a prohibited change and a scripted counterpart's actions
  are never charged to the agent.
- **Attempted and committed violations never merge.** Realised harm and blocked
  intent are different phenomena, kept apart at state, trace, verdict and metric
  level, with no Phase-1 training weight for the former.
- **Verification is offline.** Given a recorded action sequence, replay
  reproduces every state hash, responder transition and verdict with no model
  and no network — and detects an altered action, a doctored metric, a changed
  scenario or an incompatible grading version.

## Live evaluation

**Not run. Status: `BLOCKED_EXTERNAL`.**

`AnthropicClient` refuses to construct unless *both* `CERL_LIVE_EVAL_AUTHORIZED=1`
and a positive `CERL_LIVE_EVAL_BUDGET_CENTS` are set. API credentials being
present in the environment is deliberately not sufficient: a key proves you
*can* call the API, not that anyone agreed to be billed.

The integration is complete and tested offline against fixtures that are stamped
`synthetic`. No fixture, scripted policy, or oracle transcript in this
repository is a model result, and none may be reported as one.

## Documentation

| | |
|---|---|
| [`docs/design.md`](docs/design.md) | Full design of record |
| [`docs/hypotheses.md`](docs/hypotheses.md) | Preregistered claims C1–C6 |
| [`docs/enforcement.md`](docs/enforcement.md) | Layers A/B/C; attempted vs committed |
| [`docs/responders.md`](docs/responders.md) | Scripted counterparts, v1 semantics |
| [`docs/reproducibility.md`](docs/reproducibility.md) | R1–R6; the two reproducibility claims |
| [`docs/pairing.md`](docs/pairing.md) | Counterfactual pairing and Criterion 41 |
| [`docs/criterion-41.md`](docs/criterion-41.md) | Criterion 41 in detail, per family |
| [`docs/families.md`](docs/families.md) | W1, W2, W3 specifications as built |
| [`docs/limitations.md`](docs/limitations.md) | What this does not measure |
| [`docs/deviations.md`](docs/deviations.md) | Every accepted deviation, with reasons |
| [`CLAUDE.md`](CLAUDE.md) | Non-negotiable engineering rules |
| [`PROGRESS.md`](PROGRESS.md) | Current state and next action |

All data is synthetic. Licensed under Apache-2.0.
