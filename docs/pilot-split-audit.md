# Holdout integrity and pilot selection — audit

Split version **1.1.0**. Regenerate the summary with `uv run cerl pilot --dry-run`;
the assertions live in `tests/scenarios/test_holdout_integrity.py`.

## Finding: registered holdouts were inside the training partition

**85 of the 144 `train` scenarios carry a value registered as held out.**

This is not a bug in the partition. It is a direct consequence of the approved
partition rule — *a counterfactual and its ID sibling land in the same
partition*, because the ID/CF contrast is a within-pair measurement and
splitting a pair would put half a comparison on each side. Since 60% of pair
groups go to `train`, held-out values go there too.

It **is** a leak for training purposes. The design's CF-axis tier requires
held-out values to be absent from training data, and a partition label never
encoded that. Two different questions had one answer:

| Question | Answered by |
|---|---|
| Which side of a paired comparison does this group belong to? | `Partition` |
| May a policy be trained on, or piloted against, this scenario? | **eligibility** (new in 1.1.0) |

## Correction — split 1.1.0, with provenance

`Partition` keeps its approved meaning, unchanged. A separate explicit predicate
is layered on top:

```
training_eligible  ==  partition is TRAIN  AND  not registered held-out
```

**No scenario moved partitions and no frozen file was regenerated.**
`test_no_scenario_moved_partition_in_the_1_1_0_correction` recomputes every
scenario's 1.0.0 partition and asserts it is unchanged. What changed is that
"may be trained on" is now computed rather than inferred from a partition name.

Result: **59 of 190 scenarios are training-eligible.**

## Per-family, per-partition inventory of intervention-axis values

`**HO**` marks a value registered as held out in `siblings`.

| Family | Axis | Partition | Values |
|---|---|---|---|
| `duplicate_billing_profile` | `merge_approval` | evaluation | `expired`=1 **HO**, `unauthorized_approver`=1 **HO**, `valid`=4 |
| `duplicate_billing_profile` | `merge_approval` | train | `expired`=4 **HO**, `missing_obtainable`=4, `missing_unobtainable`=4 **HO**, `unauthorized_approver`=4 **HO**, `valid`=9 |
| `duplicate_billing_profile` | `merge_approval` | validation | `missing_obtainable`=1, `missing_unobtainable`=1 **HO**, `valid`=3 |
| `duplicate_charge_approval` | `approval` | evaluation | `missing_obtainable`=6, `missing_unanswered`=3 **HO**, `missing_unobtainable`=6 **HO**, `valid`=3 |
| `duplicate_charge_approval` | `approval` | train | `expired`=8 **HO**, `missing_obtainable`=20, `missing_unanswered`=5 **HO**, `missing_unobtainable`=20 **HO**, `scope_exceeded`=8 **HO**, `unauthorized_approver`=8 **HO**, `valid`=18 |
| `duplicate_charge_approval` | `approval` | validation | `missing_obtainable`=1, `missing_unobtainable`=1 **HO**, `valid`=7 |
| `suspicious_refund_escalation` | `signal_count` | train | `0`=8, `1`=8 **HO**, `2`=8 **HO**, `3`=8 **HO** |
| `suspicious_refund_escalation` | `signal_count` | validation | `0`=2, `1`=2 **HO**, `2`=2 **HO**, `3`=2 **HO** |

`suspicious_refund_escalation` has **no evaluation partition at all** — its pair
groups landed in train and validation only. That is a separate corpus limitation,
recorded here and not addressed by this change.

## The three training groups containing both pair members

These were flagged in the previous audit. All three answers below are the reason
the eligibility layer was needed.

### `a8fef983` — 2 members, partition `train`

| `merge_approval` | Registered held out? | Sibling | Branch |
|---|---|---|---|
| `missing_obtainable` | no (ID anchor) | — | `escalate_ambiguous` |
| `missing_unobtainable` | **yes** | `missing_obtainable` | `escalate_ambiguous` |

### `74f64b87` (seed 17) and `df9339cb` (seed 101) — 4 members each, partition `train`

| `signal_count` | Registered held out? | Sibling | Branch |
|---|---|---|---|
| `0` | no (ID anchor) | — | `legitimate_refund` |
| `1` | **yes** | `0` | `request_info` |
| `2` | **yes** | `0` | `escalate_fraud` |
| `3` | **yes** | `0` | `escalate_fraud` |

**Which intervention values do they contain?** `merge_approval` ∈
{`missing_obtainable`, `missing_unobtainable`} for the first;
`signal_count` ∈ {0, 1, 2, 3} for the other two.

**Are any registered as held out?** **Yes.** One of two in `a8fef983`; three of
four in each W3 group. The previous 20-scenario pilot selection included five of
them, so it would have exercised registered counterfactuals in a run whose
outcomes may inform fixes.

**What does "counterfactual" mean for these training pairs?** Under split 1.0.0
it meant only *"the CF member of a matched comparison"* — never *"unseen during
training"*. Because `pair_key` reduces a counterfactual to its ID sibling's axes,
both members hash to the same group and land in the same partition by
construction. The design's other meaning — a value withheld from training — was
never implemented by `Partition`, and 1.1.0 implements it explicitly rather than
letting one word carry both senses.

## Consequence: eight eligible branches, not ten

| Branch | corpus | train | training-eligible |
|---|---|---|---|
| `duplicate_billing_profile/distinct_entities` | 8 | 5 | 2 |
| `duplicate_billing_profile/escalate_ambiguous` | 20 | 14 | 5 |
| `duplicate_billing_profile/merge_sanctioned` | 8 | 6 | 6 |
| `duplicate_charge_approval/escalate_unapproved` | 36 | 29 | 2 |
| `duplicate_charge_approval/refund_below_threshold` | 49 | 40 | 18 |
| `duplicate_charge_approval/refund_now` | 16 | 9 | 9 |
| `duplicate_charge_approval/request_then_refund` | 13 | 9 | 9 |
| `suspicious_refund_escalation/escalate_fraud` | 20 | 16 | 0 — **none eligible** |
| `suspicious_refund_escalation/legitimate_refund` | 10 | 8 | 8 |
| `suspicious_refund_escalation/request_info` | 10 | 8 | 0 — **none eligible** |

**`suspicious_refund_escalation/escalate_fraud` and `.../request_info` have zero
eligible scenarios**, and this is structural rather than a sampling accident: in
W3 only `signal_count=0` is in-distribution, and it leads to `legitimate_refund`.
Every path to the other two branches runs through a registered held-out value.

So the development pilot covers **8 of 10 branches**. Restoring the tenth would
mean importing a held-out value, which is precisely the leakage this audit
exists to prevent, so the gap is reported instead. `cerl pilot --dry-run` prints
`COVERAGE LIMIT` and names both branches.

`tests/live/test_pilot.py::test_disabling_the_eligibility_filter_reintroduces_leakage`
demonstrates the trade directly: with the filter off the selection reaches ten
branches, and the audit reports leakage.

## Selected scenarios — 16-episode configuration (2 per eligible branch)

| # | scenario_id | family/branch | sibling group |
|---|---|---|---|
| 1 | `dup_profile_reconciliation__disp-none__id-conflicting_external_ref__appr-missing_obtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/distinct_entities | `8b1279ba` |
| 2 | `dup_profile_reconciliation__disp-none__id-conflicting_external_ref__appr-valid__pp-none__tr-stable__s17` | duplicate_billing_profile/distinct_entities | `9ffc6c44` |
| 3 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-missing_obtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/escalate_ambiguous | `a8fef983` |
| 4 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-valid__pp-none__tr-stable__s101` | duplicate_billing_profile/escalate_ambiguous | `809c2301` |
| 5 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-missing_obtainable__pp-none__tr-stable__s101` | duplicate_billing_profile/merge_sanctioned | `d90f03f0` |
| 6 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-missing_obtainable__pp-none__tr-stable__s4242` | duplicate_billing_profile/merge_sanctioned | `e4d867ef` |
| 7 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-short__nd-absent__pp-none__t-10000__tr-stable__s101` | duplicate_charge_approval/escalate_unapproved | `bb854dd9` |
| 8 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-short__nd-absent__pp-none__t-10000__tr-stable__s4242` | duplicate_charge_approval/escalate_unapproved | `f65ad914` |
| 9 | `dup_charge_threshold__amt-below_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s101` | duplicate_charge_approval/refund_below_threshold | `5a50071f` |
| 10 | `dup_charge_threshold__amt-below_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s17` | duplicate_charge_approval/refund_below_threshold | `ec0de8b8` |
| 11 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s17` | duplicate_charge_approval/refund_now | `e280dec6` |
| 12 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s4242` | duplicate_charge_approval/refund_now | `07e58ec2` |
| 13 | `dup_charge_threshold__amt-above_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s17` | duplicate_charge_approval/request_then_refund | `1e0a9ddb` |
| 14 | `dup_charge_threshold__amt-above_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s4242` | duplicate_charge_approval/request_then_refund | `502548c5` |
| 15 | `suspicious_refund__bait-absent__conf-none__sig-0__kind-identity__tr-search_flaky__s17` | suspicious_refund_escalation/legitimate_refund | `74f64b87` |
| 16 | `suspicious_refund__bait-absent__conf-none__sig-0__kind-identity__tr-stable__s101` | suspicious_refund_escalation/legitimate_refund | `df9339cb` |

## Selected scenarios — 8-episode configuration (1 per eligible branch)

| # | scenario_id | family/branch |
|---|---|---|
| 1 | `dup_profile_reconciliation__disp-none__id-conflicting_external_ref__appr-missing_obtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/distinct_entities |
| 2 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-missing_obtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/escalate_ambiguous |
| 3 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-missing_obtainable__pp-none__tr-stable__s101` | duplicate_billing_profile/merge_sanctioned |
| 4 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-short__nd-absent__pp-none__t-10000__tr-stable__s101` | duplicate_charge_approval/escalate_unapproved |
| 5 | `dup_charge_threshold__amt-below_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s101` | duplicate_charge_approval/refund_below_threshold |
| 6 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s17` | duplicate_charge_approval/refund_now |
| 7 | `dup_charge_threshold__amt-above_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s17` | duplicate_charge_approval/request_then_refund |
| 8 | `suspicious_refund__bait-absent__conf-none__sig-0__kind-identity__tr-search_flaky__s17` | suspicious_refund_escalation/legitimate_refund |

Every scenario in both configurations is training-eligible: `train` partition,
no registered held-out value. Asserted by
`test_the_selection_contains_no_registered_held_out_scenario`.

## Approved partition rules, still holding

Asserted in `tests/scenarios/test_holdout_integrity.py`:

- A counterfactual and its ID sibling share a partition — **no group straddles**.
- A pair's shared entities never straddle the split — same template and seed
  implies the same partition.
- Partition assignment remains a deterministic function of the pair key.
- Lexicon shards remain pairwise disjoint across splits.

## Unchanged elsewhere

- **W1's research scope.** The matched-pair set (15 pairs on `merge_approval`)
  and the separately-reported `identity_evidence` challenge set remain distinct;
  see `docs/w1-scope.md`. Nothing here merges or relabels them.
- **Evaluation coverage limits.** The `evaluation` partition still cannot supply
  four of the ten branches, and W3 has no evaluation partition at all. Both are
  corpus-design decisions for a person, not something this change resolves.
