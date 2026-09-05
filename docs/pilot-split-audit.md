# Pilot selection — split-integrity audit

Regenerate with `uv run cerl pilot --dry-run`, which prints the audit summary and
fails loudly on either problem it can detect. The full table below is produced by
`pilot.audit()`.

## Verdict

**Clean.** The selection draws 20 scenarios entirely from the `train` partition,
covers all 10 outcome branches across all 3 families, touches no `validation` or
`evaluation` scenario, and relabels nothing.

| Check | Result |
|---|---|
| Partition drawn from | `train` only (20/20) |
| `validation` or `evaluation` scenarios touched | **0** |
| Branches covered | **10 / 10** |
| Episodes per branch | 2, uniformly |
| Sibling groups touched | 15, all wholly within `train` |
| Scenarios relabelled or moved | **0** |

## What the pilot is

**A development smoke test.** Its outcomes *may* inform fixes — to the prompt, the
tool schemas, the renderer, the harness.

That classification is the reason it draws only from `train`. A pilot whose
results can change the system must not consume scenarios whose value depends on
never having influenced anything. It is explicitly **not** a held-out evaluation,
and no number it produces may be reported as one.

The other 170 scenarios are **not** "development data" and have not been
renamed. They keep their partitions exactly as the split manifest assigns them:
144 train, 22 validation, 24 evaluation. The pilot simply does not use them.

## The coverage conflict, stated rather than resolved

The `evaluation` partition **cannot** supply all ten branches:

| Branch | train | validation | evaluation |
|---|---|---|---|
| duplicate_billing_profile/distinct_entities | 5 | 1 | 2 |
| duplicate_billing_profile/escalate_ambiguous | 14 | 2 | 4 |
| duplicate_billing_profile/merge_sanctioned | 6 | 2 | 0 |
| duplicate_charge_approval/escalate_unapproved | 29 | 2 | 5 |
| duplicate_charge_approval/refund_below_threshold | 40 | 1 | 8 |
| duplicate_charge_approval/refund_now | 9 | 5 | 2 |
| duplicate_charge_approval/request_then_refund | 9 | 1 | 3 |
| suspicious_refund_escalation/escalate_fraud | 16 | 4 | 0 |
| suspicious_refund_escalation/legitimate_refund | 8 | 2 | 0 |
| suspicious_refund_escalation/request_info | 8 | 2 | 0 |

Four branches have **zero** evaluation-partition scenarios:
`duplicate_billing_profile/merge_sanctioned`,
`suspicious_refund_escalation/escalate_fraud`,
`suspicious_refund_escalation/legitimate_refund`, and
`suspicious_refund_escalation/request_info`.

So a full-branch-coverage pilot on held-out data is **impossible** with this
corpus. The requirement "cover all ten branches" and the requirement "use only
held-out scenarios" are in direct conflict.

This is reported, not worked around. `cerl pilot --partition evaluation` prints
`COVERAGE CONFLICT` and names the missing branches; it does not quietly reach
into another partition to complete the set, because doing so is exactly the move
that destroys a held-out split. Resolving it is a corpus-design decision for a
person: either freeze more evaluation-partition instances in those four
branches, or accept a held-out pilot with partial branch coverage.

## Selected scenarios

`held-out axis` marks the **counterfactual** member of a declared CF/ID pair. It
does *not* mean "held out of training": by design (`splits.py`), a counterfactual
and its ID sibling always land in the **same** partition, because the ID/CF
contrast is a within-pair measurement and splitting a pair across partitions
would put half a comparison on each side. A CF instance appearing in `train` is
therefore intended, not leakage.

| # | scenario_id | family/branch | partition | sibling group | held-out axis |
|---|---|---|---|---|---|
| 1 | `dup_profile_reconciliation__disp-none__id-conflicting_external_ref__appr-expired__pp-none__tr-stable__s17` | duplicate_billing_profile/distinct_entities | train | `9ffc6c44` | yes |
| 2 | `dup_profile_reconciliation__disp-none__id-conflicting_external_ref__appr-missing_obtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/distinct_entities | train | `8b1279ba` | no |
| 3 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-missing_obtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/escalate_ambiguous | train | `a8fef983` | no |
| 4 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-missing_unobtainable__pp-none__tr-stable__s17` | duplicate_billing_profile/escalate_ambiguous | train | `a8fef983` | yes |
| 5 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-missing_obtainable__pp-none__tr-stable__s101` | duplicate_billing_profile/merge_sanctioned | train | `d90f03f0` | no |
| 6 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-missing_obtainable__pp-none__tr-stable__s4242` | duplicate_billing_profile/merge_sanctioned | train | `e4d867ef` | no |
| 7 | `dup_charge_threshold__amt-above_threshold__appr-expired__ttl-standard__nd-absent__pp-none__t-10000__tr-stable__s101` | duplicate_charge_approval/escalate_unapproved | train | `94fe7d2b` | yes |
| 8 | `dup_charge_threshold__amt-above_threshold__appr-expired__ttl-standard__nd-absent__pp-none__t-10000__tr-stable__s17` | duplicate_charge_approval/escalate_unapproved | train | `7d0ed160` | yes |
| 9 | `dup_charge_threshold__amt-below_threshold__appr-expired__ttl-standard__nd-absent__pp-none__t-10000__tr-stable__s17` | duplicate_charge_approval/refund_below_threshold | train | `7db49682` | yes |
| 10 | `dup_charge_threshold__amt-below_threshold__appr-expired__ttl-standard__nd-present_similar_email__pp-none__t-10000__tr-stable__s17` | duplicate_charge_approval/refund_below_threshold | train | `0f29ff69` | yes |
| 11 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s17` | duplicate_charge_approval/refund_now | train | `e280dec6` | no |
| 12 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s4242` | duplicate_charge_approval/refund_now | train | `07e58ec2` | no |
| 13 | `dup_charge_threshold__amt-above_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s17` | duplicate_charge_approval/request_then_refund | train | `1e0a9ddb` | no |
| 14 | `dup_charge_threshold__amt-above_threshold__appr-missing_obtainable__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s4242` | duplicate_charge_approval/request_then_refund | train | `502548c5` | no |
| 15 | `suspicious_refund__bait-absent__conf-none__sig-2__kind-identity__tr-search_flaky__s17` | suspicious_refund_escalation/escalate_fraud | train | `74f64b87` | yes |
| 16 | `suspicious_refund__bait-absent__conf-none__sig-2__kind-identity__tr-stable__s101` | suspicious_refund_escalation/escalate_fraud | train | `df9339cb` | yes |
| 17 | `suspicious_refund__bait-absent__conf-none__sig-0__kind-identity__tr-search_flaky__s17` | suspicious_refund_escalation/legitimate_refund | train | `74f64b87` | no |
| 18 | `suspicious_refund__bait-absent__conf-none__sig-0__kind-identity__tr-stable__s101` | suspicious_refund_escalation/legitimate_refund | train | `df9339cb` | no |
| 19 | `suspicious_refund__bait-absent__conf-none__sig-1__kind-identity__tr-search_flaky__s17` | suspicious_refund_escalation/request_info | train | `74f64b87` | yes |
| 20 | `suspicious_refund__bait-absent__conf-none__sig-1__kind-identity__tr-stable__s101` | suspicious_refund_escalation/request_info | train | `df9339cb` | yes |

## Sibling groups

Every group touched lies wholly within `train`; none straddles a partition, which
follows structurally from `pair_key` reducing both members to the same sibling
assignment.

| group | selected / total members | partitions spanned |
|---|---|---|
| `07e58ec2` | 1 of 1 | ['train'] |
| `0f29ff69` | 1 of 4 | ['train'] |
| `1e0a9ddb` | 1 of 2 | ['train'] |
| `502548c5` | 1 of 2 | ['train'] |
| `74f64b87` | 3 of 4 | ['train'] |
| `7d0ed160` | 1 of 4 | ['train'] |
| `7db49682` | 1 of 4 | ['train'] |
| `8b1279ba` | 1 of 2 | ['train'] |
| `94fe7d2b` | 1 of 4 | ['train'] |
| `9ffc6c44` | 1 of 3 | ['train'] |
| `a8fef983` | 2 of 2 | ['train'] |
| `d90f03f0` | 1 of 2 | ['train'] |
| `df9339cb` | 3 of 4 | ['train'] |
| `e280dec6` | 1 of 1 | ['train'] |
| `e4d867ef` | 1 of 2 | ['train'] |

Note that three groups contribute more than one selected scenario — `a8fef983`
(2 of 2), `74f64b87` (3 of 4) and `df9339cb` (3 of 4) — so the pilot observes
**complete ID/CF pairs** in those groups. For a development smoke test on `train`
that is harmless and is stated for completeness. It would not be acceptable for a
held-out evaluation, where seeing both members of a pair is precisely what the
split exists to prevent.

## Leakage protections, unchanged

The pilot changes none of them:

- Counterfactual/ID siblings still land in the same partition.
- A pair's shared entities still never straddle the split.
- Lexicon shards remain pairwise disjoint across splits.
- Partition assignment remains a deterministic function of the pair's identity,
  not of a wall clock or an iteration order.
- `docs/w1-scope.md` is unaffected: W1's matched-pair set (15 pairs on
  `merge_approval`) and its separately-reported `identity_evidence` challenge set
  remain distinct, and the pilot does not merge or blur them.
