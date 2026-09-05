# CERL-Bench progress

Working file. Updated at each meaningful milestone and before context compaction.
After compaction: reload `CLAUDE.md`, `docs/design.md`, and this file, then continue
from **Next action**.

## Authorized scope

Phase 1B, authorized directly by the user in-session on 2026-09-05. Supersedes the
earlier "stop after Phase 1A" and "stop after one small W1 slice" instructions.

**In scope**: W1 and W3 families, scenario generation/splits/pairing across all three
families, evaluation harness + offline replay, unprivileged prompt-only baseline
(integration only unless a spending budget exists), documentation, verification.

**Out of scope**: push/publish/deploy, history rewrite, frontend, HTTP/MCP adapters,
SFT, GRPO, later phases, paid infrastructure. No Chrome, no ChatGPT during this run.

## Baseline verified at start

`b7fc987698119ea1bae3036671284d344029b888` — confirmed by inspection, not assumed:
114 frozen W2 scenarios, 114 gold trajectories, clean tree at that commit.

## Milestones

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

## Interrupted work preserved (uncommitted at resume)

Mid-flight W1 groundwork, all intentional:

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

`scenario/bootstrap.py` imports a W1 module that does not exist yet, so the package
does not import until that lands. That is the immediate next action.

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

All authorised Phase 1B work is complete and verified from a clean clone of
`7904d9f`. See `MORNING_REPORT.md`.

The next useful step is Phase 2 (scale to ~40 templates, pilot variance study,
prompt-only results across all four tiers), which requires an explicit spending
budget before any live model run.

**Live model evaluation is BLOCKED_EXTERNAL.** `ANTHROPIC_API_KEY` and
`GEMINI_API_KEY` are present in the environment, but no spending budget or
provider authorization is recorded anywhere in the project or session, and
credentials alone are explicitly not a budget. `AnthropicClient` refuses to
construct unless both `CERL_LIVE_EVAL_AUTHORIZED=1` and a positive
`CERL_LIVE_EVAL_BUDGET_CENTS` are set. No API call was made during this run.
Every baseline fixture is stamped `synthetic` and none may be reported as a
model result.

## Evidence locations

- Frozen scenarios: `scenarios/frozen/`, manifest `scenarios/manifest.json`
- Gold trajectories: `scenarios/gold/`
- Golden verdicts: `tests/golden/golden_verdicts.json`
- Reports: `MORNING_REPORT.md` (written at handoff)
