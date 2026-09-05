# Canonical split 1.2.0 — leakage repair and audit

Rebuild and inspect with:

```bash
uv run cerl splits                 # partition sizes, canonical version
uv run cerl splits --show-moves    # every sibling group that changed partition
uv run cerl splits --write         # regenerate scenarios/split_manifest.json
```

Assertions read the canonical split directly, not the pilot selector:
`tests/scenarios/test_canonical_split.py`.

## Why 1.1.0 was not a fix

1.1.0 added a runtime **eligibility filter** inside the pilot selector. The pilot
was clean; the split underneath it was not. `partition_of` still returned `train`
for 85 registered counterfactuals, so any other consumer — a future training loop
above all — would have read them as training data.

A filter in one consumer is not a fixed split. 1.2.0 repairs the split itself,
and the filter is retained only as a backstop against a future regression. That
it is now inert is the clearest evidence the repair is real:
`test_disabling_the_eligibility_filter_is_now_inert` shows the filtered and raw
selections are identical.

## Version history — all three preserved

| Version | Status | Rule |
|---|---|---|
| **1.0.0** | frozen historical, still computable via `splits.partition_v1_0_0` | 60/15/25 hash of the pair key. Pair-integral, Criterion 42 violating |
| **1.1.0** | frozen historical, `training_ineligibility` retained as a backstop | 1.0.0 plus a runtime eligibility filter at the selector |
| **1.2.0** | **canonical** | a sibling group containing any registered held-out value is never training data |

Neither 1.0.0 nor 1.1.0 was rewritten. `partition_v1_0_0` is asserted verbatim by
`test_the_1_0_0_rule_is_preserved_verbatim`, so results recorded against either
stay interpretable and the migration is auditable in both directions.

## The 1.2.0 rule

```
group is CF-bearing  ==  any member carries a registered held-out value

CF-bearing  and 1.0.0 said train  ->  moved whole, to validation or evaluation
CF-bearing  and 1.0.0 said other  ->  unchanged
pure-ID                            ->  1.0.0 assignment preserved exactly
```

Groups move **whole**, so pair integrity survives. Which of validation or
evaluation a moved group goes to is `int(pair_key[8:16], 16) % 100 < 38`,
preserving the 15:25 ratio 1.0.0 used between them.

## The five guarantees

| # | Guarantee | Status | Evidence |
|---|---|---|---|
| 1 | No training scenario contains any registered held-out axis value | **HOLDS** | `test_the_canonical_training_partition_contains_no_registered_holdout`; 0 of 15 |
| 2 | Lexicon shards pairwise disjoint across train/validation/evaluation | **FAILS** | see below — unfixable by relabelling |
| 3 | Every ID/CF sibling pair stays in the same partition | **HOLDS** | `test_every_sibling_group_is_in_exactly_one_partition`; 0 straddling |
| 4 | No scenario silently omitted, relabelled, or duplicated | **HOLDS** | `verify_manifest_totality` returns no problems; 190 ids, no duplicates |
| 5 | Held-out values namespaced by family and axis | **HOLDS** | `family/axis=value`; W1 and W2 share value names, so the namespace is load-bearing |

### Guarantee 2 fails, and cannot be repaired by moving groups

**Every frozen scenario draws from the `core` lexicon shard.** Partitions share
15–17 of ~17 customer names. The shard machinery exists and its pools *are*
pairwise disjoint; the corpus was generated before it was wired in.

Moving groups between partitions cannot fix this — the names are baked into the
frozen files, and every partition would still be drawing from `core`. The fix is
to regenerate the corpus against per-partition shards, which changes every
scenario hash and requires a generator version bump. That is a corpus change,
outside the scope of a split repair that was explicitly asked to preserve hashes.

`test_the_corpus_does_not_yet_use_disjoint_shards_across_partitions` asserts the
current state deliberately, and fails loudly with instructions once the corpus is
regenerated.

## Groups moved

| Move | Groups | Scenarios |
|---|---|---|
| train → evaluation | 28 | 80 |
| train → validation | 16 | 49 |
| **total moved** | **44** | **129** |

44 of 86 sibling groups changed partition; 42 kept their 1.0.0 assignment.
`uv run cerl splits --show-moves` lists every one with its reason, e.g.

```
0ad417ed  train -> evaluation  (4 members)  moved out of training: group carries
          registered held-out duplicate_charge_approval/approval=expired,
          duplicate_charge_approval/approval=scope_exceeded,
          duplicate_charge_approval/approval=unauthorized_approver
```

**No scenario file was regenerated.** Every committed sha256 in
`scenarios/manifest.json` still matches the bytes on disk
(`test_no_scenario_file_changed_in_the_migration`), and seeds, generator
versions and provenance are untouched. The split manifest carries labels only —
it deliberately duplicates no scenario provenance.

## Scenario counts by family and partition

| Family | train | validation | evaluation | total |
|---|---|---|---|---|
| `duplicate_billing_profile` | 5 | 13 | 18 | 36 |
| `duplicate_charge_approval` | 10 | 34 | 70 | 114 |
| `suspicious_refund_escalation` | 0 | 24 | 16 | 40 |
| **total** | **15** | **71** | **104** | **190** |

Compare 1.0.0: 144 / 22 / 24. Training fell from 144 to 15 because 159 of the
190 scenarios live in groups that contain a registered counterfactual.

## Branch coverage by partition

| Branch | train | validation | evaluation |
|---|---|---|---|
| `duplicate_billing_profile/distinct_entities` | 0 — **none** | 1 | 7 |
| `duplicate_billing_profile/escalate_ambiguous` | 4 | 7 | 9 |
| `duplicate_billing_profile/merge_sanctioned` | 1 | 5 | 2 |
| `duplicate_charge_approval/escalate_unapproved` | 2 | 10 | 24 |
| `duplicate_charge_approval/refund_below_threshold` | 4 | 13 | 32 |
| `duplicate_charge_approval/refund_now` | 4 | 6 | 6 |
| `duplicate_charge_approval/request_then_refund` | 0 — **none** | 5 | 8 |
| `suspicious_refund_escalation/escalate_fraud` | 0 — **none** | 12 | 8 |
| `suspicious_refund_escalation/legitimate_refund` | 0 — **none** | 6 | 4 |
| `suspicious_refund_escalation/request_info` | 0 — **none** | 6 | 4 |
| **branches covered** | **5/10** | **10/10** | **10/10** |

**Training coverage fell from 8 branches to 5.** This is reported, not repaired:
restoring any of the five would mean importing a held-out value.

- `suspicious_refund_escalation` contributes **no training scenario at all**.
  Every W3 branch needs a `signal_count` value and only `0` is in-distribution;
  its pure-ID groups all landed outside training under the unchanged 1.0.0 hash.
- `duplicate_charge_approval/request_then_refund` and
  `duplicate_billing_profile/distinct_entities` lost their last training members
  the same way.

**The evaluation-coverage limitation is resolved as a side effect.** Under 1.0.0
the evaluation partition could supply only 6 of 10 branches and W3 had no
evaluation partition at all. Under 1.2.0 both validation and evaluation cover
**10/10**. This was not engineered for — it follows from moving CF-bearing groups
out of training — and it is the more important half of the corpus for measuring
generalization.

## Held-out values by partition

| Registered value | train | validation | evaluation |
|---|---|---|---|
| `duplicate_billing_profile/merge_approval=expired` | 0 | 2 | 3 |
| `duplicate_billing_profile/merge_approval=missing_unobtainable` | 0 | 2 | 3 |
| `duplicate_billing_profile/merge_approval=unauthorized_approver` | 0 | 2 | 3 |
| `duplicate_charge_approval/approval=expired` | 0 | 3 | 5 |
| `duplicate_charge_approval/approval=missing_unanswered` | 0 | 1 | 7 |
| `duplicate_charge_approval/approval=missing_unobtainable` | 0 | 7 | 20 |
| `duplicate_charge_approval/approval=scope_exceeded` | 0 | 3 | 5 |
| `duplicate_charge_approval/approval=unauthorized_approver` | 0 | 3 | 5 |
| `suspicious_refund_escalation/signal_count=1` | 0 | 6 | 4 |
| `suspicious_refund_escalation/signal_count=2` | 0 | 6 | 4 |
| `suspicious_refund_escalation/signal_count=3` | 0 | 6 | 4 |
| **total held-out scenarios** | **0** | **41** | **63** |

All 11 registered held-out values are namespaced `family/axis=value`. W1's
`merge_approval` and W2's `approval` share the value names `expired` and
`missing_unobtainable`, so an unnamespaced label would silently conflate two
different workflows' counterfactuals.

## Sibling-pair integrity

| Check | Result |
|---|---|
| Groups straddling a partition | **0** of 86 |
| CF scenarios whose ID sibling is in another partition | **0** |
| Scenarios in more than one group | **0** |
| Corpus scenarios missing from the manifest | **0** |
| Manifest entries not in the corpus | **0** |
| Committed manifest vs. fresh rebuild | identical |
| Manifest hash-pinned | yes (`manifest_hash`) |

## Lexicon-shard disjointness

| Check | Result |
|---|---|
| Shard *pools* pairwise disjoint | **yes** — `core`, `eval_a`, `eval_b` share no name |
| Corpus *partitions* draw disjoint names | **NO** — all scenarios use `core` |
| train ∩ validation | 17 shared names |
| train ∩ evaluation | 17 shared names |
| validation ∩ evaluation | 15 shared names |

## Pilot under the corrected split

| Config | Episodes | Branches | Estimated | Worst case | Cap |
|---|---|---|---|---|---|
| 1/branch | 5 | 5/10 | $1.08 | $12.82 | $13 |
| 2/branch | 9 | 5/10 | $1.91 | $23.11 | $24 |

**The 8-episode, 8-branch pilot did not survive the repair and has not been
preserved.** The honest result is **5 episodes across 5 branches**, one per
eligible training branch, cap **$13**. The 2-per-branch configuration yields 9,
not 10, because one eligible branch has a single training scenario.

### Selected scenarios — 5-episode development pilot

| # | scenario_id | family/branch |
|---|---|---|
| 1 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-valid__pp-none__tr-stable__s101` | duplicate_billing_profile/escalate_ambiguous |
| 2 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-valid__pp-none__tr-search_flaky__s17` | duplicate_billing_profile/merge_sanctioned |
| 3 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-short__nd-absent__pp-none__t-10000__tr-stable__s101` | duplicate_charge_approval/escalate_unapproved |
| 4 | `dup_charge_threshold__amt-below_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s101` | duplicate_charge_approval/refund_below_threshold |
| 5 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s17` | duplicate_charge_approval/refund_now |

All five are in the canonical training partition and carry no registered
held-out value.

## Criterion 42 — acceptance

> *42. Lexicon shards pairwise disjoint across splits; no frozen training-split
> instance contains a held-out axis value.*

Two clauses, and they do not have the same answer.

| Clause | Verdict | Evidence |
|---|---|---|
| No held-out value in the training split | **PASS** | 0 of 15 training scenarios; asserted on the canonical split, twice, including across every intervention axis |
| Lexicon shards pairwise disjoint across splits | **FAIL** | all scenarios use `core`; partitions share 15–17 names |

**Criterion 42 overall: FAIL.** It may be marked PASS only when both clauses
hold literally, and the first does not. The remedy is a corpus regeneration
against per-partition shards with a generator version bump — a scenario-authoring
change, not a split change, and outside this repair's scope.

## Unchanged

- **W1's research scope.** The matched-pair set (15 pairs on `merge_approval`)
  and the separately-reported `identity_evidence` challenge set stay distinct;
  see `docs/w1-scope.md`. No axis value was invented, no trajectory padded.
- **Criterion 41.** 104 matched pairs, unaffected: pairs did not change, only
  which partition they sit in.
- **Every frozen scenario file**, its hash, seed, generator version and
  provenance.
