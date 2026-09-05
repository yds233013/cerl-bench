# Live-evaluation pilot — proposal

**Status: NOT EXECUTED. No paid API call has been made.** This document is the
configuration to review and approve; nothing here runs without an explicit
authorisation step described in §7.

Every cost figure below is **measured**, not assumed: prompt sizes come from
serialising the exact request payloads the harness builds, over the actual pilot
scenarios, driven offline by the oracle. The measurement script is
`scripts/estimate_pilot_cost.py` and re-runs free.

---

## 1. Provider and model

| | |
|---|---|
| Provider | Anthropic first-party API (`anthropic` Python SDK) |
| Model id | **`claude-opus-5`** |
| Thinking | `{"type": "adaptive"}` |
| Effort | `output_config={"effort": "medium"}` |
| `max_tokens` | **2048** |
| Temperature | provider default (adaptive thinking; not pinned) |
| Prompt caching | enabled on the system + tool-schema prefix |

**Pricing used** (US$/million tokens, first-party API, cached 2026-09-05 in
`src/cerl/agents/budget.py`): input **$5.00**, output **$25.00**, cache read
$0.50, cache write $6.25. `cost_cents` raises `KeyError` on an unpriced model
rather than defaulting to zero — an unpriced model would otherwise look free and
run uncapped.

Determinism note: this run produces a **Claim 2** artifact (best-effort model
reproduction), never Claim 1. Adaptive thinking is not seed-pinned, so the run is
not byte-reproducible by re-execution. What *is* reproducible is the replay:
every episode's action sequence is recorded, and `cerl eval --verify-manifest`
reproduces every state hash, verdict, and metric offline with no model in the
loop. That is the claim the benchmark rests on.

## 2. Scenario selection — all 3 families, all 10 branches

**2 scenarios per outcome branch = 20 episodes.** Selection is deterministic:
within each `(family, branch)`, the two lowest `scenario_id` values in sort order.
No sampling, no seed, so the selection is reproducible from the corpus alone and
is fixed before any result is seen.

| Family | Branch | Episodes |
|---|---|---|
| `dup_profile_reconciliation` (W1) | `merge_sanctioned` | 2 |
| | `distinct_entities` | 2 |
| | `escalate_ambiguous` | 2 |
| `duplicate_charge_approval` (W2) | `refund_now` | 2 |
| | `request_then_refund` | 2 |
| | `refund_below_threshold` | 2 |
| | `escalate_unapproved` | 2 |
| `suspicious_refund_escalation` (W3) | `legitimate_refund` | 2 |
| | `request_info` | 2 |
| | `escalate_fraud` | 2 |
| **Total** | **10 branches** | **20** |

Two per branch is the minimum that distinguishes "the policy handles this branch"
from "one episode happened to go well". It does not support per-branch
significance testing, and this document claims none — §8.

**Development scenarios stay available.** The 170 scenarios outside this
selection remain fully usable for debugging against `ScriptedClient` and
`TranscriptCacheClient`, both offline and free. Debugging never touches the
pilot selection.

## 3. Frozen prompt

The pilot uses `SYSTEM_PROMPT` from `src/cerl/agents/prompt_only.py` and the tool
schemas from `all_tool_schemas()`, **unchanged**, at the commit recorded in the
manifest. The manifest records `renderer_version` and a hash of the system prompt
and tool schemas, so a later run with different prompt text is detectable rather
than silently comparable.

**No prompt tuning against held-out outcomes.** The prompt is frozen before the
run. If the pilot shows the prompt is defective — malformed-action loops,
systematic tool misuse — the finding is reported and any revision is a *new*
pilot with a new manifest, not an edit that makes the existing numbers look
better. Iterating a prompt against evaluation outcomes converts a held-out split
into a training signal, which would invalidate exactly the ID/CF contrast the
benchmark exists to measure.

## 4. Limits per episode

| Limit | Value | Enforced by |
|---|---|---|
| Steps | 40 (`budget_steps`) | `CerlEnv`; exceeding truncates → `INCOMPLETE` |
| Harness steps | 40 | `run_agent(max_steps=...)`, independent of the env |
| Model requests | ≤ 1 per step | one `complete()` per `act()` |
| Retries per request | ≤ 2 (3 attempts) | `AnthropicClient.max_retries` |
| `max_tokens` | 2048 | request parameter |
| Wall-clock timeout | 120 s per request | SDK default; a timeout counts as a failed attempt |

## 5. Cost — measured

Measured over the actual 20-scenario selection. "Expected" drives each episode to
its oracle-length trajectory at ~500 output tokens/turn; "worst case" drives every
episode to the full 40-step budget with every response hitting `max_tokens`.

| Config | Episodes | Expected | Worst case |
|---|---|---|---|
| A — 2/branch, 4096 tok, no caching | 20 | $8.60 | $138.99 |
| B — 2/branch, 4096 tok, caching | 20 | $4.03 | $90.32 |
| **C — 2/branch, 2048 tok, caching (proposed)** | **20** | **$4.03** | **$49.36** |
| D — 1/branch, 2048 tok, caching (fallback) | 10 | $1.99 | $24.68 |

Per-episode measured profile: 9–16 steps (median 11), mean 44.7k input tokens and
4.5k output tokens uncached across a whole episode.

**The honest part: expected and worst case differ by 12×.** Input grows
quadratically with steps — every turn re-sends the whole transcript — so an agent
that flails to the step budget costs far more than one that solves the task in
eleven steps. A single mean is not a spending bound, and no configuration makes
that gap go away.

What makes the run safe is not a tight estimate but a **hard cap that is checked
before each request** (§6). Config C is chosen because its worst case is small
enough to authorise outright: approving $50 means the run cannot cost more than
$50 even if every episode behaves as badly as the environment permits.

**Recommended spending cap: $50.00 (5000 cents).**

If $50 is more than the reviewer wants to authorise, config **D** halves the
selection to one episode per branch — still all 10 branches, worst case $24.68,
cap $25. That is the conservative configuration, and the trade is explicit: one
episode per branch cannot distinguish a branch the policy handles from a lucky
run.

## 6. How the budget is enforced *during* execution

Requiring a positive budget at client construction is a gate, not spending
control: it is checked once, before anything has been spent, and never again. A
run could exceed it on its second request and continue to its end.

Enforcement is `BudgetLedger` (`src/cerl/agents/budget.py`), wired into
`AnthropicClient.complete`:

1. **Reserve before sending.** Each request estimates its input tokens from the
   serialised payload and reserves `cost(input, max_tokens)` — the full
   `max_tokens` as output, because actual output is unknown until the response
   arrives. If `spent + reserved + estimate > cap`, the request raises
   `BudgetExceeded` and **is never sent**.
2. **Retries reserve separately.** Every attempt reserves in its own right. A
   retry is a billable request; a loop that reserves once per logical call
   underbills by up to 3× and is how a capped run overruns.
3. **Outstanding requests hold their reservation.** The reservation is deducted
   before the call and reconciled after, under a lock, so concurrent in-flight
   requests cannot each pass a check against the same unspent balance that they
   jointly fail.
4. **Settle from real usage.** On success the reservation is released and the
   *actual* cost from `response.usage` is recorded, so the worst-case estimate
   does not keep accruing.
5. **Failures release.** An attempt that returns no usage releases its
   reservation; the next attempt makes its own, checked against the cap in turn.
6. **The env cap is the ceiling.** A caller may pass a ledger with a *smaller*
   cap than `CERL_LIVE_EVAL_BUDGET_CENTS`, never a larger one — otherwise
   authorisation could be widened in code.
7. **Input estimates are biased high** (~3.2 chars/token rather than ~4).
   Under-estimating input is the one error that lets a run exceed its cap.

Covered by `tests/live/test_budget.py` (8 tests), including a 40-thread
concurrency test asserting that reservations bound simultaneous requests, and a
test that an unpriced model raises rather than costing zero.

**Residual risk, stated plainly.** The pre-flight estimate is a character-based
approximation, not `count_tokens`. It is biased high, and the cap is enforced
against the estimate, so the realistic failure mode is stopping *early*, not
overrunning. A tokenizer-exact pre-flight would require an extra API call per
request; before a run where the margin matters, switch to
`client.messages.count_tokens`.

## 7. Exact commands

Nothing below runs today: `CERL_LIVE_EVAL_AUTHORIZED` is unset, so
`AnthropicClient` construction raises `LiveEvaluationNotAuthorized`.

```bash
# 0. Free, offline. Re-derive the selection and the cost table above.
uv run cerl pilot --dry-run
uv run python scripts/estimate_pilot_cost.py

# 1. Free, offline. Prove the budget enforcement and the client wiring.
uv run pytest tests/live -q

# 2. Authorise. Two independent signals; a stray budget value in a shell
#    profile cannot by itself start spending.
export ANTHROPIC_API_KEY=...             # supplied by the operator
export CERL_LIVE_EVAL_AUTHORIZED=1
export CERL_LIVE_EVAL_BUDGET_CENTS=5000  # $50.00 cap, enforced per request

# 3. Execute. NOT IMPLEMENTED, deliberately: `--execute` reports that the
#    pilot requires reviewed authorisation and exits non-zero. Wiring a flag
#    that quietly starts spending is exactly what should not exist until this
#    proposal is approved.
uv run cerl pilot --execute        # exits 1 with an explanatory message

# 4. Free, offline. After a pilot has been run and its manifest recorded:
#    replay it with no network and no model in the loop (Claim 1).
uv run cerl verify-manifest runs/pilot_opus5.json
```

**What step 3 will need when it is approved**, and what does not yet exist: an
execution path in `cerl pilot` that builds an `AnthropicClient` with a
`BudgetLedger` capped at the authorised amount, runs `run_agent` per selected
scenario, records each episode's transcript into the cache, and writes a run
manifest with `privilege_mode: unprivileged`. The pieces it composes —
`PromptOnlyAgent`, `run_agent`, `AnthropicClient`, `BudgetLedger`,
`manifest.write` — are all built and tested offline. The transcript-cache
*replay* path (`--regenerate --from-cache`) is likewise unbuilt at the CLI level;
`TranscriptCacheClient` exists and is tested, but there is no command wired to it,
and this document does not claim one.

Step 4 is the point of the exercise: the pilot's scientific content is the
recorded action sequences, and those stay verifiable forever without spending
anything again.

## 8. What this pilot can and cannot establish

**Can:** that the prompt-only harness runs end to end against a live model; the
failure-class distribution of an untrained model across all ten branches; whether
the model produces parseable tool calls at all; a per-branch success signal
precise enough to decide whether a larger run is worth funding; and a real
transcript cache enabling free offline replay thereafter.

**Cannot:** any C1–C6 claim. 20 episodes across 10 branches, 2 per branch, with no
ID/CF split, supports no generalization-gap estimate and no confidence interval.
The cluster bootstrap in §12.4 of the design needs the full split structure and
many more episodes per template. **No result from this pilot may be reported as a
measurement of Δ.**

It is also not a baseline arm for the paper. It is a smoke test with a price tag,
and calling it more than that would be the kind of overclaim the rest of this
repository is built to prevent.
