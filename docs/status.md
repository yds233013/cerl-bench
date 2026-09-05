# Status — Phase 1B candidate

One canonical status. `README.md` and `MORNING_REPORT.md` both defer to this
file; where any other document disagrees, this one is correct.

Last updated 2026-09-05, after the Phase 1B closeout.

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
| Criterion 41 — matched CF/ID pairs | **PASS, on 104 pairs** | see deviation 1 |
| W1 `identity_evidence` generalization | **NOT TESTED** | see deviation 1 |
| Prompt-only live baseline | **BLOCKED_EXTERNAL** | see deviation 2 |
| Live-evaluation pilot | **PROPOSED, NOT EXECUTED** | `docs/live-pilot-proposal.md` |
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

`BLOCKED_EXTERNAL`. No provider spending budget has been authorised, and
credentials alone are not a budget. The client refuses to construct without
`CERL_LIVE_EVAL_AUTHORIZED=1` and a positive `CERL_LIVE_EVAL_BUDGET_CENTS`, and
spend is metered per request by `BudgetLedger`.

The integration is built and tested offline against fixtures stamped
`source="synthetic"`. **No fixture, scripted policy, or oracle transcript is
reported anywhere as a model result.** A pilot is specified in
`docs/live-pilot-proposal.md` and awaits review; nothing in it has been run.

## What this candidate does not establish

- No C1–C6 claim. Those need trained arms, the full split structure, and a
  cluster bootstrap over templates. None exists.
- No statement about any model's safety. The only policies measured are the
  privileged references and five deterministic controls.
- The two privileged reference policies are an upper bound by construction, not
  baseline arms, and are labelled as such wherever they appear.
