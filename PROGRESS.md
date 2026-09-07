# CERL-Bench progress

Working file. Updated at each meaningful milestone and before context compaction.
After compaction: reload `CLAUDE.md`, `docs/design.md`, and this file, then continue
from **Next action**.

## Authorized scope — current

**v0.1 release-candidate repair**, authorized 2026-09-07: fix the three issues
raised by independent review of candidate `ed04213`, and nothing else.

**In scope**: exact aggregate verification; closing the document-aliasing hole
without regressing the ~50 ms episode budget; repairing provenance and stale
status text; running the listed gates from a fresh clone.

**Out of scope**: new workflow families, RL training of any kind, MCP adapters,
inference, spending, or a hosted deployment.

**Publication (2026-09-07):** v0.1.0 is released at
https://github.com/yds233013/cerl-bench. Earlier "do not push" instructions in
this file were standing constraints at the time and are now superseded for the
source; they still hold for anything that would spend money or run inference.

> **Scope history.** Earlier entries in this file were written under narrower
> grants, and their "out of scope" lines are superseded rather than wrong: the
> Phase 1B grant of 2026-09-05 excluded a frontend and HTTP adapters, both of
> which were separately authorized afterwards and now exist (`cerl serve`,
> `cerl serve-review`, `app/`). `CLAUDE.md` §11 is the authority on current
> scope; where this file disagrees with it, §11 wins.

## Baseline verified at start of Phase 1B — historical

`b7fc987698119ea1bae3036671284d344029b888` — confirmed by inspection, not assumed:
114 frozen W2 scenarios, 114 gold trajectories, clean tree at that commit. This is
where Phase 1B began, not where the project stands.

## Milestones — Phase 1B

Historical: this table records the Phase 1B run and is not a current to-do list.


| # | Milestone | Status |
|---|---|---|
| 1 | Resume safely from interrupted state | DONE |
| 2 | Durable progress file | DONE (this file) |
| 3 | W2 integrity check + policy-assumption documentation | PARTIAL - code verified, docs pending |
| 4 | W1 complete (all approved branches) | DONE - 36 frozen, 3 branches, oracle clean |
| 5 | W3 complete | DONE - 40 frozen, 3 branches, C_DISCLOSE exercised |
| 6 | Scenario coverage + CF/ID pairing across families | DONE - 190 scenarios, pairing green in all 3 |
| 7 | Evaluation harness + offline replay | DONE - cerl eval / verify-manifest / splits |
| 8 | Prompt-only baseline (integration) | DONE - live runs BLOCKED_EXTERNAL |
| 9 | Documentation + executable demonstrations | DONE |
| 10 | Clean-clone verification + morning handoff | DONE |

## Interrupted work preserved (uncommitted at resume) — historical

Mid-flight W1 groundwork as it stood mid-run. **All of it landed**; this section
is kept as a record of the resume, not as outstanding work:

- `scenario/families/registry.py` (new) — family registry: template + generator + plan.
  Freezing no longer knows which family it holds.
- `scenario/bootstrap.py` (new) — one-shot family registration on package import.
- `scenario/freeze.py` — uses the registry instead of the hard-coded W2 generator.
- `verify/approval.py` — `check_approval(..., required_role)` generalised; refund and
  merge wrappers on top. Hard-coding one role would have let a refund approval
  authorise a profile merge.
- `tools/billing/handlers.py` — merge repoints invoices and payment methods too,
  per approved merge semantics (was charges only).
- `verify/predicates/{state,trace,invariant}.py` — W1 predicates added.

At the moment this was written, `scenario/bootstrap.py` imported a W1 module that
did not exist yet, so the package would not import. That was resolved in the same
run; it is recorded here because the resume protocol depended on it.

## Significant decisions

1. **Family registry rather than per-family branching in `freeze`.** A family is a
   triple (template, generator, instance plan); the freezing machinery should not
   know which one it holds. `scenario` still may not import `reference`, so the
   family→oracle mapping lives in `reference/registry.py`.
2. **Approval role is a parameter, not a constant.** Merges require
   `merge_approver`, refunds `refund_approver`.
3. **Branch-scoped invariants.** W1 needs `merge_customers` prohibited in the
   abstain/escalate branches but sanctioned in the merge branch, so `InvariantSpec`
   gains the same `branches` / `applies_when` scoping the rubric already has.
4. **W1's Criterion-41 intervention axis is `merge_approval`, not
   `identity_evidence`.** Measured first, then decided: identity evidence selects
   which *workflow* is correct (merge 12 calls / escalate 10 / distinct 9), so
   pairing across it gives |delta| of 2-3. That cost difference is real -- merging
   is irreversible and the policy demands verification that tagging does not.
   Padding the cheap branches would be the manipulation the criterion forbids, so
   identity evidence is **stratified** across splits instead: all three values on
   both sides, where no ID/CF difficulty gap can arise. Recorded as a scope
   statement, not a passed criterion.
5. **Family-specific scenario-id slugs.** Sharing W2's abbreviation map with W1
   silently dropped W1's axis names and collided 150 scenarios onto 119 files.
   Slugs are now per family and `cerl freeze` refuses to write if the distinct-id
   count does not match the instance count.

## Next action

**Nothing is pending.** The reviewed repairs are complete and the gates were run
from a fresh clone; see `docs/rc-review-repairs.md` for each finding's before and
after, and `docs/status.md` for canonical status.

Phase 2 (scale to ~40 templates, pilot variance study, prompt-only results across
all four tiers) remains the next *useful* step, and is **not authorized**. It
requires an explicit spending budget before any live model run.

**Live model evaluation is BLOCKED_EXTERNAL.** `ANTHROPIC_API_KEY` and
`GEMINI_API_KEY` are present in the environment, but no spending budget or
provider authorization is recorded anywhere in the project or session, and
credentials alone are explicitly not a budget. `AnthropicClient` refuses to
construct unless both `CERL_LIVE_EVAL_AUTHORIZED=1` and a positive
`CERL_LIVE_EVAL_BUDGET_CENTS` are set. No paid API call has ever been made.
Every baseline fixture is stamped `synthetic` and none may be reported as a
model result. The one real model run in this repository is local
(`qwen3:4b` via Ollama), recorded under `evidence/local-baseline/`.

## Evidence locations

- Frozen scenarios: `scenarios/frozen/`, manifest `scenarios/manifest.json`
- Gold trajectories: `scenarios/gold/`
- Golden verdicts: `tests/golden/golden_verdicts.json`
- Recorded runs: `evidence/`, including `evidence/rc-review/before_after.json`
- Reports: `MORNING_REPORT.md` and `LOCAL_BASELINE_REPORT.md` — both **historical**,
  each pinned to the commit it describes; `docs/status.md` is canonical for now
