# Status — Phase 1B candidate

One canonical status. `README.md` and `MORNING_REPORT.md` both defer to this
file; where any other document disagrees, this one is correct.

Last updated 2026-09-06, after the local-model baseline milestone.

## Two different kinds of "not done"

Kept apart deliberately, because they call for different responses:

| Label | Meaning | Who unblocks it |
|---|---|---|
| **`NOT_IMPLEMENTED`** | Wiring that does not exist. Implementation work. | Us |
| **`BLOCKED_EXTERNAL`** | Built and tested, waiting on something outside the repository — here, spending authorisation. | The person who authorises spend |

Reporting missing wiring as an external blocker would hide work behind a
dependency that is not actually the obstacle.

## Summary

Phase 1B is **implemented and verified, with two scope deviations recorded
below**. It is a candidate for review, not a finished result: no live model has
been evaluated, and no research claim (C1–C6) has been measured.

## By area

| Area | Status | Evidence |
|---|---|---|
| Three workflow families (W1, W2, W3), 190 frozen scenarios, 10 branches | **IMPLEMENTED** | `cerl inspect --assert-cells all` |
| Oracle scores a clean 1.0 on every frozen scenario | **VERIFIED** | 190/190, all branches |
| Alternative reference policy, materially different, also 1.0 | **VERIFIED** | 190/190 |
| Verifier fidelity — mutations produce their expected failure class | **VERIFIED** | 1,489 checks, 0 misclassifications |
| Adversarial fixtures T1–T17 | **VERIFIED** | all present and passing |
| Deterministic replay, hash-chained traces, byte-exact freeze | **VERIFIED** | identical hashes on Python 3.11 / 3.12 / 3.13 |
| Import contracts and privilege boundary | **VERIFIED** | 6/6 contracts, 4 privilege proofs |
| Evaluation harness, run manifest, offline verification | **IMPLEMENTED** | `cerl eval`, `cerl verify-manifest` |
| Control baselines (5 unprivileged, 2 privileged references) | **MEASURED** | `docs/baselines.md`, pinned in `tests/eval/test_controls.py` |
| Control metrics separated (task / harm / decision) | **IMPLEMENTED** | a single "safe" column conflated task success with harmlessness |
| Criterion 41 — matched CF/ID pairs | **PASS, on 104 pairs** | see deviation 1 |
| **Criterion 42** — lexicon disjointness **and** no held-out value in training | **PASS** | both clauses hold literally on corpus 2.0.0: 0 shared lexicon values between partitions, 0 holdouts in training |
| Corpus 2.0.0 (regenerated, partition-aware lexicon) | **IMPLEMENTED** | 190 scenarios, byte-stable re-freeze, oracle clean 190/190 |
| Canonical split 1.2.0 | **IMPLEMENTED** | 44 sibling groups moved out of training |
| W1 `identity_evidence` generalization | **NOT TESTED** | see deviation 1 |
| Pilot selection + split audit | **IMPLEMENTED** | `cerl pilot --dry-run`; `docs/pilot-split-audit.md` |
| Pilot execution path | **IMPLEMENTED** | `cerl pilot --execute --synthetic`, 20/20 episodes offline |
| Transcript recording | **IMPLEMENTED** | recorded per episode, stamped with the transport's source |
| Offline regeneration from cache | **IMPLEMENTED** | `cerl regenerate`; a cache miss fails rather than calling out |
| Offline action replay + manifest verification | **IMPLEMENTED** | `cerl verify-manifest` |
| Spend accounting (4 quantities, retries, resume) | **IMPLEMENTED** | `docs/budget-accounting.md`; 18 tests |
| **Grader comparison study (state-only vs trace-aware)** | **COMPLETE** | 1,134 cases; `docs/grader-study-results.md` |
| **Local-model agent + recorded run** | **IMPLEMENTED** | `qwen3:4b` via Ollama; replay verified; `LOCAL_BASELINE_REPORT.md` |
| Refund tool contract (`reason` vocabulary) | **FIXED** | schema advertises the closed set; tool schema 1.0.0 → **1.1.0** |
| Inference deadline, measured outside the simulator | **IMPLEMENTED** | bounded per request, nothing starts after expiry; server-side cancellation not claimed |
| Completed episode from a local model | **NOT ACHIEVED** | both attempts ended at a limit; ~6 tok/s makes a 16-turn episode ~45 min |
| **Live (paid) pilot execution** | **BLOCKED_EXTERNAL** | see deviation 2 — built, tested, awaiting authorisation |
| Per-request ledger persistence | **IMPLEMENTED** | write-ahead journal, `fsync`ed before each send; 12 crash-recovery tests |
| Holdout-vs-training inventory | **IMPLEMENTED** | `docs/pilot-split-audit.md` |
| Per-partition lexicon shards in the corpus | **IMPLEMENTED** | company names, email domains, staff handles and display names all sharded |
| Ten-branch training coverage | **BLOCKED — research decision** | 5/10. Three branches could be added by freezing pure-ID instances; two cannot without changing the hypotheses. Alternatives documented, none chosen |
| Evaluation-partition branch coverage | **RESOLVED by 1.2.0** | was 6/10; validation and evaluation now cover 10/10 |
| Investigate-then-abstain / -act controls | **NOT_IMPLEMENTED** | `docs/baselines.md` |
| SFT, GRPO, curriculum arms, HTTP/MCP adapters, frontend | **NOT STARTED** | out of Phase 1B scope by instruction |

## Deviation 1 — W1's identity-evidence axis is a challenge set, not a matched pair

Criterion 41 requires |Δ oracle tool calls| ≤ 1 between a counterfactual scenario
and its in-distribution sibling. **104 declared pairs satisfy it, with no
exceptions and no waiver.** W1 contributes 15 of those, all on the
`merge_approval` axis.

W1's `identity_evidence` axis **cannot** satisfy it. Its three values cost 12, 10
and 9 oracle tool calls, because they lead to three genuinely different
workflows. Those scenarios are preserved, reported separately with their measured
differences, and **excluded from the Criterion 41 result** — they are not counted
as passing, and nothing was padded, discarded, or waived to make them fit.

The consequence is stated rather than hidden: the benchmark does not currently
test whether a policy generalizes across identity-evidence strength in W1. Three
options for future work are in `docs/w1-scope.md` §6.

A related overclaim has been **withdrawn**: an earlier version of
`docs/pairing.md` implied that stratifying an axis across splits establishes that
its values are equally difficult. It does not. Stratification changes which
comparison is made; it does not equalise the things compared.

## Deviation 2 — no live model evaluation

**`BLOCKED_EXTERNAL`, and now genuinely so.** Previously this label covered both
a missing budget *and* missing wiring, which was not an honest use of it. The
wiring now exists: selection, execution, transcript recording, cache
regeneration, action replay, and metered spend are implemented and exercised end
to end offline. What remains missing is only the authorisation.

The client refuses to construct without `CERL_LIVE_EVAL_AUTHORIZED=1` and a
positive `CERL_LIVE_EVAL_BUDGET_CENTS`; a ledger cannot widen the authorised cap;
and `cerl pilot --execute` without both exits non-zero.

**Provenance cannot be faked.** A response's `source` comes from the transport
object that produced it, not from a flag, so the synthetic transport cannot emit
a `live` label. Synthetic runs record `verification_mode: synthetic` and
`provider: synthetic-transport` in the manifest, and the CLI prints a warning.
**No fixture, scripted policy, or oracle transcript is reported anywhere as a
model result.**

## Deviation 3 — the pilot is a development smoke test, not a held-out evaluation

The pilot draws only from the canonical training partition, because its outcomes
may inform fixes. It is not a held-out evaluation and no result from it may be
reported as one. The other 175 scenarios keep their partitions and are **not**
relabelled as development data.

Training reaches **5 of 10 branches** under split 1.2.0, so the pilot is 5
episodes. That limit is reported by `cerl pilot`, which names the five missing
branches, and it is never resolved by importing a held-out value.

## Deviation 5 — Criterion 42 now passes; the corpus was regenerated to get there

**Resolved.** Both clauses hold literally on corpus 2.0.0.

**Clause B — no held-out value in training.** Split 1.0.0 put 85 registered
counterfactuals in the training partition; 1.1.0 filtered them at the pilot
selector, which left the split broken for every other consumer; **1.2.0** moved
every CF-bearing sibling group out of training, whole. Training holds 15
scenarios and zero holdouts.

**Clause A — lexicon disjointness.** Corpus 1.x decided partitions after
generation, so every file drew from the `core` pool and partitions shared 15–17
entity names. That could not be repaired by relabelling. **Corpus 2.0.0** assigns
partitions from the plan *before* materialising anything, so each scenario is
generated from its own partition's shard. Measured over the committed files: **0
shared values** between any two partitions.

Hashes changed because the files changed — that is what the version bump is for.
Corpus 1.x remains on disk, untouched and re-verified.

**Cost, stated rather than repaired:** training coverage is **5 of 10 branches**,
and W3 contributes no training scenario. See deviation 6.

## Deviation 6 — training branch coverage is 5/10, pending a research decision

Two different causes:

- `distinct_entities`, `request_then_refund` and `legitimate_refund` are all
  reachable from in-distribution values. They are absent because of which
  instances were frozen, not what the axes permit. **Additional pure-ID instances
  at fresh seeds would add them, with no hypothesis change.**
- `request_info` and `escalate_fraud` have **no** in-distribution value —
  reachable only through registered holdouts. Adding them would mean
  re-registering W3's `signal_count` holdouts, which changes what W3's
  counterfactual is and therefore changes H0 and C1–C6.

Four alternatives are tabulated in `docs/pilot-split-audit.md`; **none is
chosen**, because the choice is a research decision about corpus design. Nothing
was invented and no scenario was moved to improve coverage.

## Deviation 5 (historical) — how Criterion 42 came to fail

Kept as the record of the two-stage failure, since both stages are instructive.
Superseded by the section above.

Criterion 42 has **two** clauses, and for a while they did not have the same
answer.

**Clause B — no held-out value in the training split: PASS.** An inventory found
85 of the 144 `train` scenarios under split 1.0.0 carried a registered held-out
value, a consequence of the approved rule that a counterfactual and its ID
sibling share a partition. Split **1.1.0** filtered them at the pilot selector,
which left the split itself broken for every other consumer. Split **1.2.0**
repairs the split: a sibling group containing any registered held-out value is
never training data, and such groups move whole so pair integrity survives. 44 of
86 groups moved; **no scenario file was regenerated** and every committed sha256
still matches. Training now holds 15 scenarios and **zero** holdouts, asserted
directly on `partition_of` rather than through the selector.

**Clause A — lexicon shards pairwise disjoint across splits: FAIL.** Every frozen
scenario draws from the `core` shard, so partitions share 15–17 entity names. The
shard pools *are* disjoint; the corpus was generated before they were wired in.
This cannot be repaired by relabelling — the names are in the frozen files — and
fixing it means regenerating the corpus with a generator version bump. Recorded
as `NOT_IMPLEMENTED`, with a test that fails loudly once it is done.

The cost of clause B is stated rather than engineered around: **training coverage
fell from 8 branches to 5**, and W3 contributes no training scenario at all. The
development pilot is correspondingly **5 episodes, not 8**. Detail, tables and
per-group provenance in `docs/pilot-split-audit.md`.

## Deviation 4 — no dollar guarantee on spend

A strict dollar guarantee cannot be established from outside the provider. What
exists is an operational limit: no request is sent unless its worst-case price
fits the remaining balance, every retry and every ambiguous outcome is charged,
and spend survives both resumption and a mid-episode crash. The earlier claim
that a character-based estimate made overspending impossible has been
**withdrawn**.

Crash safety was the fifth residual exposure and is now closed: a write-ahead
journal records each request before it is sent, recovery preserves confirmed
usage, charges outcomeless reservations as unresolved, never replays them, and
does not reset the allowance. What remains is that the journal is consistent with
what *this process observed*, never with what the provider billed —
`docs/budget-accounting.md` §Remaining limitation.

## Local-model baseline (2026-09-06)

An open-weight model (`qwen3:4b`, Apache-2.0) running under Ollama on this
machine, driving the real environment through the ordinary agent interface. Five
training-partition W2 episodes, sequential, no paid inference.

**Result: 0/5 safe task completion.** One episode committed a `C_AUTH` violation
by refunding on an expired approval — a correctly-detected safety failure with
Invariant B1 holding. The other four ran out of steps without declaring an
outcome, because **84% of turns produced no tool call**: the model exhausts its
640-token output budget on reasoning prose. Throughput was 5.8 tok/s against a
machine already 22 GB into swap.

**Diagnosed 2026-09-06.** The 84% figure conflated three outcomes; separated, it
is 71% output truncation, 12% completed-without-a-call, **0% invalid arguments
and 0% unknown tools**. Root cause was a protocol error in our client: the Qwen3
template primes a `<think>` block on every request, and `think:false` only
stopped the runner *parsing* it, so reasoning arrived as `content` and was
appended to conversation history every turn. Token cost is identical either way.
The 640-token output cap was also marginal — the simplest real turn needs 609.
With `think=true` and a 2048 cap, executed actions rose 16% → 57%.

Two fixes committed: `think=true` by default with reasoning kept out of history,
and an invalid enum argument now scored as malformed rather than crashing the
episode. **No completed episode exists at the corrected configuration** — the
bounded check was interrupted by that crash, and the 30-minute inference budget
was exhausted. The ~6 tok/s generation rate remains unexplained; the earlier
attribution to swap was overstated. Full detail: `LOCAL_BASELINE_REPORT.md` §9.

## Grader comparison study (2026-09-06)

State-only versus trace-aware grading over 1,134 constructed cases spanning 114
W2 scenarios. Protocol frozen before results; `uv run cerl grader-study` runs it
offline in ~73 s.

**Task completion: both graders 1134/1134, perfect agreement. Safety:
trace-aware 1134/1134; state-only 1020/1134 with 0 false positives and 114 false
negatives.** Every disagreement is one failure mode — a prohibited change that
was later restored, which appears in no terminal diff. State-only misses all 114
and gets everything else right.

Two defects were found and both were mine, not the graders': an annotation that
conflated task completion with decision correctness, and a baseline that blamed
the agent for a responder's own records. Originals preserved; the second was
exposed by evaluation cases and is disclosed as development-exposed.

On the six recorded model trajectories the graders **agree everywhere**,
including both `C_AUTH` detections. No naturally occurring instance of the
restored-change failure has been observed. Detail: `docs/grader-study-results.md`.

## What this candidate does not establish

- No C1–C6 claim. Those need trained arms, the full split structure, and a
  cluster bootstrap over templates. None exists.
- No statement about any model's safety. The only policies measured are the
  privileged references and five deterministic controls.
- The two privileged reference policies are an upper bound by construction, not
  baseline arms, and are labelled as such wherever they appear.
- 36/190 is what one specific escalating control scored. It is not a proven
  ceiling on escalation-shaped policies, and none is claimed.
- Nothing produced by the synthetic transport is a model result.
- The local baseline is a development smoke test on training data, n=5, one
  model, one configuration. It is not a held-out evaluation of anything.
