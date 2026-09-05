# CERL-Bench

**Counterfactual Enterprise Reinforcement Learning for Safe Tool-Using Agents**

A research benchmark and RL environment for testing whether agents that succeed
at enterprise tool-use learn *transferable safe decision procedures*, or merely
fit the state distributions they were trained on.

The core question needs an environment where a single controlled perturbation —
an approval that expired, a customer with a near-identical name, a threshold moved
from $100 to $500 — flips the correct behaviour while leaving surface complexity
nearly unchanged. Everything here follows from that.

## Status: Phase 1A

One workflow (**W2 — duplicate charge with an approval threshold**), complete end
to end. Simulator, deterministic verifier, hash-chained traces with exact replay,
scripted oracle, and 87 committed frozen scenarios across four outcome branches.

**No training, no evaluated agent, no HTTP/MCP adapters** — those are later
phases. A benchmark that makes training claims before its own determinism and
verifier fidelity are demonstrated is not worth publishing.

## Quick start

```bash
uv sync
uv run pytest -q                    # 194 tests
uv run cerl inspect --assert-cells all
uv run cerl run --scenario <id> --agent oracle --show-trace
```

## What makes it a benchmark rather than a simulator

- **The oracle scores a clean 1.0 on every frozen scenario.** Not 0.99. This
  simultaneously proves every scenario is solvable, that the branch-conditional
  rubric resolves coherently, and that the closed-world allowlist is not too
  tight for the intended solution.
- **Closed-world side-effect grading.** Anything not matched by the active
  branch's allowlist is a residual. A denylist loses to RL by construction.
- **Trace-wide, not terminal.** Violate-then-revert is caught; a revert cannot
  launder a prohibited change.
- **The environment never blocks on policy grounds** (Invariant B1). A refund
  without approval *succeeds* and latches a violation. Blocking it would prevent
  violations rather than measure them.
- **Attempted and committed violations never merge.** Realised harm and blocked
  intent are different phenomena, kept apart at state, trace, verdict and metric
  level, with no Phase-1 training weight for the former.
- **Verification is offline.** Given a recorded action sequence, replay
  reproduces every state hash and verdict with no model in the loop.

## Documentation

| | |
|---|---|
| [`docs/design.md`](docs/design.md) | Full design of record |
| [`docs/hypotheses.md`](docs/hypotheses.md) | Preregistered claims C1–C6 |
| [`docs/enforcement.md`](docs/enforcement.md) | Layers A/B/C; attempted vs committed |
| [`docs/responders.md`](docs/responders.md) | Scripted Slack counterparts, v1 semantics |
| [`docs/reproducibility.md`](docs/reproducibility.md) | R1–R6; the two reproducibility claims |
| [`docs/limitations.md`](docs/limitations.md) | What this does not measure |
| [`CLAUDE.md`](CLAUDE.md) | Non-negotiable engineering rules |

All data is synthetic. Licensed under Apache-2.0.
