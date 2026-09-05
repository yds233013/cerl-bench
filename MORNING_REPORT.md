# Phase 1B — morning handoff

**Tested code commit: `7904d9fbeff796349a8e8d8f0d70e7d5ee7fe3cb`** on branch `main`.
Everything below was run inside a fresh clone of that exact commit. Nothing was
pushed. A later commit adds this report only; the code under test is `7904d9f`.

Work window: 02:51 → 05:1x PDT (~2.4h of the authorised 10h). The run stopped
once the authorised milestones were built and verified, rather than filling the
remaining time.

**Status correction (2026-09-05 closeout).** This report originally said all
Phase 1B milestones were "complete and verified". That was too strong. Two scope
deviations are recorded in [`docs/status.md`](docs/status.md), which is now the
canonical status: W1's `identity_evidence` axis cannot satisfy Criterion 41 and
is reported as a separate challenge set (the criterion passes on 104 pairs that
exclude it), and live model evaluation remains `BLOCKED_EXTERNAL`. The
implementation and verification numbers below are unchanged and still hold.

---

## What was actually completed

| # | Milestone | Status |
|---|---|---|
| 1 | Resume safely from the interrupted state | DONE |
| 2 | Durable progress file | DONE (`PROGRESS.md`) |
| 3 | W2 integrity check + policy-assumption documentation | DONE |
| 4 | W1 complete, all approved branches | DONE |
| 5 | W3 complete, all approved branches | DONE |
| 6 | Scenario coverage + CF/ID pairing across families | DONE |
| 7 | Evaluation harness + offline replay | DONE |
| 8 | Prompt-only baseline (integration) | DONE; live runs `BLOCKED_EXTERNAL` |
| 9 | Documentation + executable demonstrations | DONE |
| 10 | Clean-clone verification + handoff | DONE |

### Commits (oldest first)

| Commit | What |
|---|---|
| `9f0c90b` | W1: duplicate billing profile reconciliation, all three branches; multi-family plumbing |
| `a9099a6` | W1 mutations, adversarial fixtures, corrected escalation taxonomy |
| `2c7d495` | W3: suspicious refund requests, with disclosure as a graded harm |
| `59f2760` | Evaluation harness: splits, run manifest, offline verification |
| `4b6251c` | Unprivileged prompt-only baseline; live evaluation `BLOCKED_EXTERNAL` |
| `7904d9f` | Documentation and executable demonstrations ← **tested commit** |

---

## Results by family and outcome branch

190 frozen scenarios. **The oracle scores a clean 1.0 on all 190.**

| Family | Scenarios | Branches (count) | Pairs | Mutations |
|---|---|---|---|---|
| `duplicate_charge_approval` (W2) | 114 | refund_now (16), request_then_refund (13), refund_below_threshold (49), escalate_unapproved (36) | 59 | 906 checks, 15 distinct |
| `duplicate_billing_profile` (W1) | 36 | merge_sanctioned (8), distinct_entities (8), escalate_ambiguous (20) | 15 | 223 checks, 12 distinct |
| `suspicious_refund_escalation` (W3) | 40 | legitimate_refund (10), request_info (10), escalate_fraud (20) | 30 | 360 checks, 11 distinct |

Every family contains at least one branch where acting is correct and one where
it is not, so **no always-escalate policy can solve every scenario in a family**.
It can still succeed on escalation cases, and the measured numbers say how far
that goes: a policy that merely *declares* escalation scores 0/190 safe (correct
*decision* on all 76 escalation scenarios, but escalating correctly also requires
posting the reference and setting the ticket status), while one that does the
escalation work reaches 36/190 — solving `escalate_unapproved` completely and no
other branch, because W1 and W3 each demand family-specific investigation first.
Full table in `docs/baselines.md`.

---

## Verification commands and exact results

All run inside a fresh clone of `7904d9f`, in that clone only.

| Command | Result | Exit |
|---|---|---|
| `uv sync` | 32 packages | 0 |
| `uv run ruff check src tests` | All checks passed! | 0 |
| `uv run mypy --strict src tests/typing` | no issues in 105 source files | 0 |
| `uv run lint-imports` | **6 contracts kept, 0 broken** | 0 |
| `uv run pytest -q` | **323 passed** | 0 |
| `uv run pytest --cov=cerl` | TOTAL 92%; core 95.0%, diff 91.2%, trace 97.0%, verify 96.5% | 0 |
| `uv run cerl freeze` | 190 scenarios, oracle clean; **0 files changed** | 0 |
| `uv run cerl inspect --assert-cells all` | all ten W2 cells at ≥3 seeds | 0 |
| `uv run cerl splits` | train 144 / validation 22 / evaluation 24 | 0 |
| `uv run cerl eval --partition evaluation` | 24 episodes, safe completion 1.000 | 0 |
| `uv run cerl verify-manifest runs/eval.json` | verified 24 episodes offline | 0 |
| `uv run cerl demo` | all five behaviours as documented | 0 |
| Vertical slice, W2 | `request_then_refund`, 5/5 rubric, SUCCESS, 14 calls | 0 |
| Vertical slice, W1 | `merge_sanctioned`, 7/7 rubric, SUCCESS, 12 calls | 0 |
| Vertical slice, W3 | `escalate_fraud`, 6/6 rubric, SUCCESS, 9 calls | 0 |

**Freeze reproduces committed artifacts byte-exactly** — `git status` reported
zero changed files after re-freezing all 190 scenarios in the clone.

**Offline replay was verified with the socket layer disabled**, so "no network"
is enforced rather than asserted.

### Measurements

- Oracle clean 1.0: **190/190**
- Mutation checks: **1,489 across three families, 0 misclassifications**, 38 distinct mutations
- Alternative policy correct and materially different: **190/190**
- Criterion 41: **104/104 pairs pass, max |Δ| = 1**
- Cross-version determinism: digest `eb44a180…` identical on Python 3.11 / 3.12 / 3.13
- Episode time: **26.0 ms** (budget 50 ms)

---

## Phase 1A regression status

**No regressions.** All Phase 1A properties still hold and are still tested:
27 action kinds / 23 tools; Invariant B1 (the environment never blocks on policy
grounds); closed-world trace-wide grading partitioned by origin; attempted and
committed violations separate at four levels with no training weight; responder
no-chaining and one-transition-per-step with declared-only exemptions; deep
immutability and validated nominal identifiers; the four privilege proofs;
byte-exact frozen regeneration; the 31-symbol ambient-nondeterminism guard.

W2's own numbers are unchanged: 114 scenarios, 59 pairs, all passing.

---

## Phase 1B acceptance checklist

| Criterion | Status | Evidence |
|---|---|---|
| W1 implemented, all approved branches | **PASS** | 36 scenarios, 3 branches, oracle 36/36 clean |
| W1 sanctioned merge semantics + prohibited cases | **PASS** | `test_w1_fidelity.py`; tombstone-not-delete, canonical direction, dispute interlock |
| W1: confusable profile cannot be merged as correct | **PASS** | `test_merging_a_similarly_named_different_customer_is_not_a_correct_reconciliation` |
| W1 two materially different correct trajectories | **PASS** | 36/36 |
| W3 implemented, all approved branches | **PASS** | 40 scenarios, 3 branches, oracle 40/40 clean |
| W3 fraud signals from observable evidence only | **PASS** | signals recomputed from each frozen world reproduce the branch count |
| W3 disclosure graded, all branches | **PASS** | `test_disclosure_is_forbidden_in_every_branch_including_the_legitimate_one` |
| W3 two materially different correct trajectories | **PASS** | 40/40 |
| No always-escalate policy solves a whole family | **PASS** | per-family tests; measured ceiling 36/190, `docs/baselines.md` |
| Deterministic, inspectable ID sibling mapping | **PASS** | `siblings.py`; freeze plan closed under it |
| Pairs differ only in the intervention axis | **PASS** | `test_pairs_differ_only_in_one_declared_intervention_axis` |
| Equal entity-set cardinality | **PASS** | 104/104 |
| Brief-length difference ≤ 15% | **PASS** | 104/104 |
| Reference tool-call difference ≤ 1 | **PASS** | 104/104, max Δ = 1 |
| No padding, no branch exemptions, no omitted pairs | **PASS** | anti-padding test; totality test |
| Pairing reported for both reference policies | **PASS** | `test_criterion_41_also_holds_under_the_alternative_policy` |
| Registered difficulty metric identified | **PASS** | the oracle; `test_the_registered_metric_is_the_oracle` |
| Explicit train/validation/evaluation partitions | **PASS** | `cerl splits`; 144/22/24 |
| Matched siblings in the same partition | **PASS** | `test_a_pair_never_straddles_the_split` |
| Shared entities do not leak across the split | **PASS** | `test_shared_entities_within_a_pair_do_not_leak_across_the_split` |
| Seeds, generator versions, provenance, hashes preserved | **PASS** | manifest + byte-exact re-freeze |
| Evaluation harness records the required fields | **PASS** | `test_manifest_records_everything_required` |
| Offline replay without model or network | **PASS** | verified with sockets disabled |
| Transcript cache as a separate capability | **PASS** | tested independently; miss raises |
| Integrity tests detect tampering | **PASS** | altered action, doctored verdict, doctored metrics, changed scenario, changed initial state, renamed responder, version skew, missing scenario |
| Prompt-only agent interface, unprivileged | **PASS** | arity-checked; import-graph walk |
| Bounded episodes, retries, config recording, transcripts | **PASS** | `test_episode_length_is_bounded` |
| Live model evaluation | **BLOCKED_EXTERNAL** | no spending budget authorised; see below |
| Authentic transcript cache from a real model | **BLOCKED_EXTERNAL** | depends on a live run |
| Documentation complete | **PASS** | 10 docs; every README command executed |
| Executable demonstrations (5) | **PASS** | `cerl demo`, each pinned by a test |
| Clean-checkout verification | **PASS** | fresh clone of `7904d9f` |
| Phase 1A regression | **PASS** | full suite green |

**Phase 1B is not complete**: two criteria are `BLOCKED_EXTERNAL`. Every other
authorised criterion passes.

---

## Real model evaluation: not run

`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` are present in the environment. **No
spending budget or provider authorization exists anywhere in the project or
session**, and credentials alone are explicitly not a budget, so no API call was
made.

`AnthropicClient` refuses to construct unless *both* `CERL_LIVE_EVAL_AUTHORIZED=1`
and a positive `CERL_LIVE_EVAL_BUDGET_CENTS` are set — two independent signals,
so a stray value in a shell profile cannot by itself start billing. Configuration
recorded for a future live run: `claude-opus-5`, adaptive thinking, effort
`medium`, `max_tokens` 4096, bounded episodes.

**Every fixture in the test suite is stamped `synthetic`.** No fixture, scripted
policy, or oracle transcript in this repository is a model result, and none is
reported as one anywhere.

---

## Significant decisions

1. **Family registry.** A family is a triple (template, generator, instance
   plan) plus a slug; the freezing machinery no longer knows which one it holds.
   Reference policies map to families in `reference/registry.py`, because
   `scenario` may not import `reference`.
2. **W1's Criterion-41 intervention axis is `merge_approval`, not
   `identity_evidence`** — measured first, then decided. Identity evidence
   selects which workflow is correct (12/10/9 calls), so pairing across it gives
   |Δ| of 2–3. Padding the cheap branches is what the criterion forbids, so it
   is stratified across splits instead, where no ID/CF difficulty gap can arise.
   Recorded as a scope statement, not a passed criterion (`docs/pairing.md`).
3. **The W2 reauthorization policy is a synthetic benchmark assumption**, not a
   universal business rule: request when no approval exists, escalate directly
   when one exists but is invalid. Its rationale is workflow-independent
   (re-requesting against a known-bad authorisation only delays the customer)
   and it is stated as an assumption in `docs/criterion-41.md`.
4. **Branch-scoped invariants**, because merging is sanctioned in one W1 branch
   and prohibited in the others.
5. **Approval role is a parameter**, so a refund approval cannot authorise a
   merge.
6. **The agent harness lives in `eval`, not `agents`** — the import contract
   caught that constructing a `CerlEnv` gives its importer a transitive path to
   the verifier.

## Defects found and fixed during the run

Recorded because each was a real bug, not a test artifact:

1. **Allowlist `value_in` did not resolve `$.` variables**, so a legitimate merge
   diff read as a prohibited side effect. W2 never exercised it.
2. **Scenario-id slugs used W2's abbreviation map for every family**, dropping
   W1's axis names and collapsing 150 scenarios onto 119 filenames. Slugs are
   now per family, and `freeze` refuses to write on an id collision.
3. **The failure classifier treated escalation as over-escalation only when the
   required decision was "act"**, so escalating out of an abstain branch was
   reported as the opposite failure.
4. **W3 fraud signals were not independent** — a differing email domain also
   implies "not the account contact", so `signal_count=1` built a world with two.
5. **W3's velocity signal changed entity counts**, making counterfactuals
   structurally larger than their siblings. The signal now varies *when* the
   refund history happened.
6. **The committed/attempted collapse detector had two false positives** —
   `"violations"` matched inside `attempted_violations`, and it read its own
   docstrings. Re-verified afterwards against a deliberately collapsing probe.
7. **A CLI test froze over the real scenario directories**, replacing the
   committed corpus with a partial one. Freeze tests are now confined to tmp
   paths, and the manifest count caught it.

## Limitations

- Three families, one domain. Cross-*domain* transfer is untested.
- `C_DISCLOSE` is checked against a fixed list of sensitive literals; a
  paraphrase would evade it. It is a floor, not a proof of non-disclosure.
- Import contracts prove a module was not imported; they are not a security
  isolation proof. Observation scans check rendered text against a curated token
  list. Both are defence in depth.
- W1's `identity_evidence` cannot serve as a counterfactual axis, narrowing what
  W1 measures.
- No hypothesis has been tested; no experimental finding exists.

## Unfinished changes

None. The working tree is clean at `7904d9f`; every change is committed.

---

## The next task, and how to resume

Nothing authorised remains. The next useful step is **Phase 2**: scale to ~40
templates, run the pilot variance study, and produce prompt-only results across
all four evaluation tiers — which requires an explicit spending budget first.

To resume:

```bash
cd <repo> && uv sync
cat PROGRESS.md            # authorised scope, decisions, next action
uv run pytest -q           # 323 tests
uv run cerl freeze         # must report 0 changed files
uv run cerl demo           # the five behaviours
```

To enable live evaluation once a budget is authorised:

```bash
export CERL_LIVE_EVAL_AUTHORIZED=1
export CERL_LIVE_EVAL_BUDGET_CENTS=<amount>
uv sync --extra live
```

Before any training arm, freeze and hash-pin the verifier and scenario corpus —
that ordering commitment is what keeps the preregistration meaningful.
