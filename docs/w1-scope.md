# W1 research scope: what is measured, and what is not

Reconciles the implemented W1 counterfactual axis against the approved W1
specification and the preregistered hypotheses. Written because the
implementation narrowed what W1 can measure, and that narrowing must be visible
rather than absorbed.

## Summary

W1 ships **two disjoint sets** with different evidential status:

| Set | Axis | Instances | Status |
|---|---|---|---|
| **Matched-pair set** | `merge_approval` | 15 pairs (30 instances) | Satisfies Criterion 41; contributes to C1–C3 |
| **Challenge set** | `identity_evidence` | 36 instances across 3 values | **Does not satisfy Criterion 41.** Reported separately with its measured difficulty differences. Contributes to no matched-pair claim. |

The two are never combined into one number.

## 1. What the matched `merge_approval` pairs measure

The intervention holds the workflow shape fixed and varies **authorisation
only**. Both members of a pair investigate the same two profiles with the same
identity evidence, check the same disputes, and read the same policy; they
differ in whether a valid merge approval exists or can be obtained.

| Held-out CF value | ID sibling | Oracle calls (CF / ID) | \|Δ\| |
|---|---|---|---|
| `expired` | `valid` | 12 / 12 | 0 |
| `unauthorized_approver` | `valid` | 12 / 12 | 0 |
| `missing_unobtainable` | `missing_obtainable` | 14 / 15 | 1 |

15 pairs, all passing, equal entity cardinality, identical brief lengths.

**The claim these support**: whether a policy that has learned "merge when
authorised" generalises to authorisation failures it did not see in training —
an expired approval, an approver without the role, a refusal. That is a genuine
counterfactual on a fixed workflow, and it is the W1 contribution to C1, C2 and
C3.

## 2. What `identity_evidence` generalization remains UNTESTED

`identity_evidence` decides **whether merging is the right action at all**, and
is the axis the W1 specification's core hazard actually turns on: two records
that look like the same entity but may not be. Held out, it would test whether a
policy that has only ever seen verifiable duplicates correctly *declines* to
merge when the evidence is conflicting or weak.

**That question is not tested by anything in this repository.** Specifically,
none of the following has any evidence behind it:

- whether a policy trained on `strong_match` merges recognise `conflicting_external_ref` and abstain;
- whether it recognises `name_only_similarity` and escalates instead of guessing;
- whether merge-abstention transfers from authorisation failures to identity failures at all.

The scenarios exist and are scored, so a policy *can* be measured on them. What
cannot be computed is a **matched ID/CF gap**, because the difficulty is not
matched (§3). Any number produced across identity values mixes generalization
with difficulty and must be reported as a challenge-set result, never as a
Criterion 41 result.

## 3. The measured difficulty differences

Reference: `strong_match`, `merge_approval=valid`, no dispute, stable tools,
seed 17. Oracle is the registered metric; the alternative policy is shown to
confirm the gap is a property of the scenario, not of one trajectory.

| identity_evidence | Branch | Oracle calls | Alternative calls | \|Δ\| vs `strong_match` |
|---|---|---|---|---|
| `strong_match` | `merge_sanctioned` | **12** | 11 | — (reference) |
| `name_only_similarity` | `escalate_ambiguous` | **10** | 11 | **2** — fails ±1 |
| `conflicting_external_ref` | `distinct_entities` | **9** | 8 | **3** — fails ±1 |

Entity cardinality and brief length *are* matched across all three (13
customers, 7 charges, 192-character brief), so the difference is purely
interaction cost.

The difference is real and not an artifact. Merging is irreversible, so the
policy requires a dispute check and an approver-role check that recording two
records as verifiably distinct does not. Padding the cheaper branches to close
the gap is exactly the manipulation Criterion 41 forbids, and widening the
tolerance would gut the control.

## 4. Stratification: what it does and does not establish

`identity_evidence` values appear in both the training and evaluation
partitions.

**What that does establish**: no *held-out* ID/CF contrast is computed on this
axis, so no Criterion 41 pair is silently formed across mismatched difficulties,
and no reported gap on this axis can be confounded by them.

**What that does NOT establish — and any claim to the contrary is withdrawn**:
stratification does not make the three identity values equally difficult. They
demonstrably are not: 12 vs 10 vs 9 oracle calls, on three different workflows
with different irreducible verification requirements. Stratification changes
*which comparison is being made*; it does not equalise the things compared. An
aggregate score across identity values is a weighted average over unequal
sub-problems, and its movement between two policies can reflect a change in the
mix as easily as a change in capability.

An earlier version of `docs/pairing.md` stated that "an axis whose values are
present on both sides cannot produce an ID/CF difficulty gap, which is the
confound C6 exists to exclude." That was too strong — it conflated *not
computing a gap* with *there being no difficulty difference* — and has been
corrected.

## 5. Original acceptance requirements: satisfied, changed, unresolved

Against the approved W1 specification (`docs/design.md` §7.6) and Phase 1B
criteria.

### Satisfied

| Requirement | Evidence |
|---|---|
| All approved outcome branches | `merge_sanctioned`, `distinct_entities`, `escalate_ambiguous`; 36 scenarios |
| Sanctioned merge semantics | Repoints charges/invoices/payment methods to the older canonical record; tombstones rather than deletes |
| Prohibited merge cases | Wrong direction, unverified pair, unapproved, under dispute — each a measurable violation |
| Identity, billing and dispute checks | `dispute_checked` rubric item; `billing.get_dispute` required before merging |
| Conditional rubrics, branch-scoped permitted changes | Resolved at freeze time; `distinct_entities` permits only the metadata marker |
| Deterministic transitions | Byte-exact replay; identical hashes on 3.11/3.12/3.13 |
| Explicit responder declarations | Per-branch, guard-precise reachability |
| Privileged reference policy | `W1Oracle`, clean 1.0 on 36/36 |
| Two materially different correct trajectories | 36/36 under the stated rule |
| Systematic mutations and adversarial fixtures | 12 mutations, 223 checks, 0 misclassifications |
| Frozen scenarios and manifest entries | 36, byte-exact regeneration |
| Confusable profile cannot be merged as correct | Dedicated test; `C_ENTITY` + `WRONG_ENTITY` |
| An always-escalate policy cannot solve the family | Measured: 0/36 on W1 (see `docs/baselines.md`) |

### Changed

| Original intent | As implemented | Why |
|---|---|---|
| `identity_evidence` as the natural counterfactual axis | `merge_approval` is the Criterion 41 axis; `identity_evidence` is a separate challenge set | Identity evidence selects the workflow, so pairing across it fails ±1 by 2–3 calls (§3) |
| One intervention axis per family (implicit) | Pairing is declared per family and may name several axes; W1 declares one | Generalising the pairing machinery was needed for three families |

### Unresolved

| Open item | Status |
|---|---|
| Identity-evidence generalization | **Untested.** No matched-pair design exists for it. Options in §6. |
| Whether merge-abstention transfers from authorisation to identity failures | **Untested.** Requires the above. |
| Whether Criterion 41 is the right instrument for a workflow-selecting axis | **Open design question.** The criterion assumes a fixed workflow perturbed by an intervention; an axis that changes which workflow is correct may need a different control. |

## 6. Options for testing identity-evidence generalization later

Recorded so the gap is actionable, not merely acknowledged. **None is
implemented, and none should be adopted without review.**

1. **Within-branch matched pairs.** Construct counterfactuals *inside* the
   `merge_sanctioned` branch — e.g. `strong_match` vs. a near-miss that is still
   verifiable — so the workflow is fixed and only the evidence strength varies.
   Narrower than the original intent but properly matched.
2. **Difficulty-normalised reporting.** Report per-branch scores against the
   per-branch oracle cost rather than a single aggregate, so a mix shift cannot
   masquerade as a capability change. Weaker than a matched pair; it controls
   for difficulty in the analysis rather than the design.
3. **Accept the gap and preregister it.** Report identity-evidence results with
   the measured 12/10/9 costs stated alongside, and preregister that the
   comparison is confounded by difficulty. Honest, and the weakest of the three.

Option 1 is the only one that would let identity evidence contribute to C1–C3 on
the same footing as `merge_approval`.
