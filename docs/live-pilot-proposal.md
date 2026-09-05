# Live-evaluation pilot — proposal

**Status: IMPLEMENTED, NOT EXECUTED.** The execution path is built and tested
end to end against a synthetic transport. **No paid API call has been made from
this repository.** Execution awaits explicit spending authorisation.

Every cost figure below is an **estimate** derived from serialised request
payloads driven by the oracle. None is a measured live cost, and none may be
reported as one. Re-derive them free with `uv run cerl pilot --dry-run`.

---

## 1. Provider and model

| | |
|---|---|
| Provider | Anthropic first-party API (`anthropic` SDK, an optional extra) |
| Selection pool | canonical training partition (split 1.2.0) |
| Model id | **`claude-opus-5`** |
| Thinking | `{"type": "adaptive"}` |
| Effort | `output_config={"effort": "medium"}` |
| `max_tokens` | **2048** (thinking and response share this allowance) |
| SDK automatic retries | **disabled** (`max_retries=0`) — see §6 |
| Application retries | 2 (3 attempts per request) |
| Request timeout | 120 s |
| Prompt caching | enabled on the system + tool-schema prefix |
| Execution | **sequential**, one request outstanding at a time |

Pricing (US$/Mtok, cached 2026-09-05 in `src/cerl/agents/budget.py`): input
$5.00, output $25.00, cache write $6.25, cache read $0.50. An unpriced model
raises rather than costing zero.

This produces a **Claim 2** artifact (best-effort model reproduction), never
Claim 1. Adaptive thinking is not seed-pinned, so re-execution is not
byte-reproducible. What *is* reproducible offline is the recorded run: `cerl
verify-manifest` replays the actions with no model in the loop, and `cerl
regenerate` replays the transcripts from cache.

## 2. Scenario selection — audited against the canonical split

**Drawn from the canonical training partition (split 1.2.0).** Selection is
deterministic — within each `(family, branch)`, the lowest `scenario_id` values
in sort order. No sampling, no seed.

**This shrank twice, both times after finding leakage.** Split 1.0.0 put 85
registered counterfactuals in training; 1.1.0 filtered them at the selector,
which left the split itself broken for every other consumer; 1.2.0 repaired the
split by moving every CF-bearing sibling group out of training, whole. Training
is now 15 scenarios. Full migration record: `docs/pilot-split-audit.md`.

**Coverage is 5 of 10 branches.** `suspicious_refund_escalation` contributes no
training scenario at all — every W3 branch needs a `signal_count` value and only
`0` is in-distribution — and W2's `request_then_refund` and W1's
`distinct_entities` have no training members either. Restoring any of them would
mean importing a held-out value, so the gap is reported and `cerl pilot` names
the five missing branches.

| Check | Result |
|---|---|
| Pool | canonical training partition, split 1.2.0 |
| Registered held-out scenarios selected | **0** |
| Branches covered | **5 / 10** (structural limit, reported) |
| Sibling groups | none straddling a partition |
| Scenario files regenerated | **0** |

## 3. Frozen prompt

`SYSTEM_PROMPT` from `src/cerl/agents/prompt_only.py` and the schemas from
`all_tool_schemas()`, unchanged, at the commit recorded in the manifest, which
also records `renderer_version`.

**No prompt tuning against held-out outcomes.** The pilot runs on `train`, so
even its intended use — informing fixes — cannot contaminate a held-out split.
If it reveals a prompt defect, a revision is a new pilot with a new manifest, not
an edit that makes existing numbers look better.

## 4. Limits per episode

| Limit | Value | Enforced by |
|---|---|---|
| Env steps | 40 | `CerlEnv`; exceeding truncates → `INCOMPLETE` |
| Harness steps | 40 | `run_agent(max_steps=…)` |
| Model requests | ≤ 1 per step | one `complete()` per `act()` |
| Attempts per request | 3 | `AnthropicClient.max_retries=2` |
| `max_tokens` | 2048 | request parameter |
| Timeout | 120 s | transport; a timeout is charged as unresolved |

## 5. Cost — estimated, both configurations

Estimated from the real payloads over the actual `train` selection.
"Estimated" drives each episode to oracle length at ~500 output tokens/turn;
"worst case" drives every episode to its full 40-step budget with every response
at `max_tokens`.

| Config | Episodes | Branches | Estimated | Worst case | Cap |
|---|---|---|---|---|---|
| **B — 1/branch, caching (recommended first run)** | **5** | **5/10** | **$1.08** | **$12.82** | **$13** |
| A — 2/branch, caching | 9 | 5/10 | $2.00 | $23.82 | $24 |

**The 8-episode, 8-branch configuration did not survive the 1.2.0 split repair
and has not been preserved.** The honest result is **5 episodes across 5
branches**. The 2-per-branch configuration yields 9 rather than 10, because one
eligible branch has a single training scenario — and it is not padded to 10.

**Recommended: config B, cap $13** for the first paid run: one episode per
eligible training branch at the lowest exposure.

One episode per branch cannot distinguish a branch the policy handles from a
lucky run, and B claims no more than that. Neither configuration says anything
about the five branches with no training scenario.

**Estimated and worst case differ by ~12×.** Input grows quadratically with steps
because every turn re-sends the transcript, so an agent that flails to the step
budget costs far more than one that finishes in eleven steps. No configuration
removes that gap, and a single mean is not a spending bound. What makes the run
safe is the cap enforced per request, not the accuracy of the estimate.

## 6. Budget enforcement — and what it does and does not guarantee

Full treatment in **`docs/budget-accounting.md`**. In brief:

Before every request, including every retry, the client counts the **entire**
request (system prompt, full conversation history, tool schemas), inflates it by
the counter's uncertainty margin, prices it at the full `max_tokens` as output,
and **refuses to send it** if it does not fit in the spendable balance.

Four quantities are tracked separately and never collapsed:

| | Meaning | Can decrease? |
|---|---|---|
| **confirmed** | Billed usage from a response. Fact. | No |
| **reserved** | Held for a request in flight. | Yes, on a definite outcome |
| **unresolved** | Held **permanently** for a request that may have reached the provider but whose usage we never learned. | **Never** |
| **estimated** | A projection. Not money. | n/a |

Accounted for: the full request contents; thinking tokens (billed as output
within `max_tokens`, so already covered by the worst-case reservation); cache
read/write pricing modifiers; application retries; **SDK automatic retries**,
disabled at the transport so no request bypasses the ledger; and timeouts, which
are charged as unresolved rather than released. Resumption reloads prior spend,
so a restarted run cannot grant itself the cap again; it refuses a changed cap or
a different model.

**The withdrawn claim.** An earlier version said a character-based estimate
biased high meant overspending could not happen. That was wrong. A heuristic is
not a bound, and it says nothing about requests whose outcome is unknown.

**What can honestly be claimed:** an operational limit around $51, not a
contractual ceiling. Five residual exposures are enumerated in
`docs/budget-accounting.md` §Residual exposure — the largest being tokenization
error while the SDK is absent and the heuristic counter is in use. Anyone
authorising this run should also set a provider-side spending limit, which is the
only actual ceiling available.

## 7. Exact commands

### Offline and free — runnable right now

```bash
# Selection, split audit, and cost projection. Spends nothing.
uv run cerl pilot --dry-run
uv run cerl pilot --dry-run --partition evaluation   # shows the coverage conflict
uv run python scripts/estimate_pilot_cost.py

# Exercise the ENTIRE execution path offline through the synthetic transport.
# Writes a manifest, a transcript cache, and a spend ledger. Results are
# labelled synthetic and cannot be labelled otherwise.
uv run cerl pilot --execute --synthetic \
    --out runs/synthetic.json \
    --transcripts-out runs/synthetic_transcripts.json \
    --ledger-out runs/synthetic_ledger.json

# Replay that run from its transcript cache (model reproducibility, Claim 2).
uv run cerl regenerate runs/synthetic.json \
    --transcripts runs/synthetic_transcripts.json

# Replay its actions with no model at all (environment determinism, Claim 1).
uv run cerl verify-manifest runs/synthetic.json

# The tests behind all of the above.
uv run pytest tests/live -q
```

### Paid — requires authorisation that does not currently exist

```bash
export ANTHROPIC_API_KEY=...              # supplied by the operator
export CERL_LIVE_EVAL_AUTHORIZED=1        # two independent signals, so a
export CERL_LIVE_EVAL_BUDGET_CENTS=1300   # stray budget cannot start spending

# Config B: 5 episodes, one per eligible training branch, cap $13.
uv run cerl pilot --execute --per-branch 1 \
    --out runs/pilot_opus5.json \
    --transcripts-out runs/pilot_transcripts.json \
    --ledger-out runs/pilot_ledger.json

# Then, free and offline, on the real recorded run:
uv run cerl verify-manifest runs/pilot_opus5.json
uv run cerl regenerate runs/pilot_opus5.json --transcripts runs/pilot_transcripts.json
```

Without both environment variables the second block exits non-zero with
"Credentials alone are not a budget." `--dry-run` is the default; `--execute`
never runs by omission.

## 8. What this pilot can and cannot establish

**Can:** that the harness runs end to end against a live model; the
failure-class distribution of an untrained model across all ten branches;
whether the model emits parseable tool calls; a per-branch signal precise enough
to decide whether a larger run is worth funding; and a real transcript cache
enabling free offline replay thereafter.

**Cannot:** any C1–C6 claim. 5 episodes across 5 branches, drawn from training
scenarios that by construction contain no counterfactual, support no
generalization-gap estimate and no confidence interval. **No result from this
pilot may be reported as a measurement of Δ**, and because it excludes every
registered counterfactual it is not an evaluation of generalization at all. It
says nothing about the five branches with no training scenario, and nothing about
W3, which has none.

It is a smoke test with a price tag. Calling it more would be the overclaim the
rest of this repository is built to prevent.
