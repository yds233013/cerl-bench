# Status — Phase 1B candidate

One canonical status. `README.md` and `MORNING_REPORT.md` both defer to this
file; where any other document disagrees, this one is correct.

Last updated 2026-09-05, after the pilot implementation.

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
| W1 `identity_evidence` generalization | **NOT TESTED** | see deviation 1 |
| Pilot selection + split audit | **IMPLEMENTED** | `cerl pilot --dry-run`; `docs/pilot-split-audit.md` |
| Pilot execution path | **IMPLEMENTED** | `cerl pilot --execute --synthetic`, 20/20 episodes offline |
| Transcript recording | **IMPLEMENTED** | recorded per episode, stamped with the transport's source |
| Offline regeneration from cache | **IMPLEMENTED** | `cerl regenerate`; a cache miss fails rather than calling out |
| Offline action replay + manifest verification | **IMPLEMENTED** | `cerl verify-manifest` |
| Spend accounting (4 quantities, retries, resume) | **IMPLEMENTED** | `docs/budget-accounting.md`; 18 tests |
| **Live pilot execution** | **BLOCKED_EXTERNAL** | see deviation 2 — built, tested, awaiting authorisation |
| Per-request ledger persistence | **NOT_IMPLEMENTED** | saved per episode; ≤1 episode of spend can be lost from the record |
| Evaluation-partition pilot with full branch coverage | **NOT_IMPLEMENTED** (corpus) | 4 branches have no evaluation instances; conflict reported, not resolved |
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

The pilot draws only from the `train` partition, because its outcomes may inform
fixes. It is not a held-out evaluation and no result from it may be reported as
one. The other 170 scenarios keep their partitions and are **not** relabelled as
development data.

One conflict is recorded rather than resolved: the `evaluation` partition cannot
supply all ten branches — four have zero evaluation instances — so a
full-coverage held-out pilot is impossible with this corpus.
`cerl pilot --partition evaluation` reports the conflict instead of borrowing
from another partition. `docs/pilot-split-audit.md` has the detail.

## Deviation 4 — no dollar guarantee on spend

A strict dollar guarantee cannot be established from outside the provider. What
exists is an operational limit: no request is sent unless its worst-case price
fits the remaining balance, every retry and every ambiguous outcome is charged,
and spend survives resumption. Five residual exposures are enumerated in
`docs/budget-accounting.md`. The earlier claim that a character-based estimate
made overspending impossible has been **withdrawn**.

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
