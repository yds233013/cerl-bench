# Counterfactual pairing and Criterion 41

## Why the invariant exists

Criterion 41 requires each counterfactual instance's oracle tool-call count to
be within ±1 of its in-distribution sibling. It is the structural half of
control C6: if counterfactual variants are simply *harder*, then
Δ = SafeCompletion_ID − SafeCompletion_CF measures difficulty rather than
overfitting, and every downstream claim is confounded.

## The pairing is total, deterministic and inspectable

`cerl/scenario/siblings.py` declares, per family, one or more **intervention
axes**. Within a pair exactly one intervention axis differs; every other axis
and the seed are identical, so both worlds contain the same entities with the
same names and differ only in the counterfactual under test.

A vague "its ID sibling" would invite the comparison to be chosen after the fact
to suit the numbers, so the mapping is executable and the freeze plan is
**closed under it** — no held-out instance can escape the check by lacking a
partner.

| Family | Intervention axis | Held-out → ID sibling |
|---|---|---|
| W2 | `approval` | expired / unauthorized_approver / scope_exceeded → valid; missing_unobtainable / missing_unanswered → missing_obtainable |
| W1 | `merge_approval` | expired / unauthorized_approver → valid; missing_unobtainable → missing_obtainable |
| W3 | `signal_count` | 1 / 2 / 3 → 0 |

The counterpart is always the ID value sharing the same *interaction structure*
— whether an approval record exists at episode start, which decides whether the
workflow involves a request at all. Pairing `missing_unobtainable` with `valid`
would compare "ask and be refused" against "read an approval already there": a
difference in workflow shape, not in the counterfactual.

## One axis is stratified rather than held out

W1's `identity_evidence` was the obvious intervention candidate and was
**measured before being rejected**. It selects which *workflow* is correct, and
the three workflows have genuinely different irreducible costs:

| identity_evidence | branch | oracle calls |
|---|---|---|
| `strong_match` | merge_sanctioned | 12 |
| `name_only_similarity` | escalate_ambiguous | 10 |
| `conflicting_external_ref` | distinct_entities | 9 |

Pairing across those gives |Δ| of 2 and 3. The cost difference is real — merging
is irreversible and the policy demands dispute and approval verification that
recording two records as distinct does not. Padding the cheaper branches would
be exactly the manipulation the criterion forbids, and widening the tolerance
would gut the control.

So `identity_evidence` is **stratified**: all three values appear in both the
training and evaluation partitions, and it is reported as a separate **challenge
set** rather than as matched pairs.

**Stratification does not establish equal difficulty**, and no claim here should
be read as saying it does. The three values demonstrably differ — 12 / 10 / 9
oracle calls on three different workflows. What stratification does is ensure no
held-out ID/CF contrast is computed on this axis, so no Criterion 41 pair is
formed across mismatched difficulties. It changes which comparison is made; it
does not equalise the things compared.

The consequence is that `identity_evidence` **contributes to no Criterion 41
result and to no matched-pair claim**, and its scenarios are reported separately
with their measured difficulty differences. See `docs/w1-scope.md` for the full
reconciliation, including what identity-evidence generalization remains
untested.

## What is enforced

`tests/scenarios/test_pairing.py`, over every family:

- every held-out instance has exactly one declared, frozen ID sibling;
- pairs differ in one intervention axis and nothing else, seed included;
- |Δ oracle tool calls| ≤ 1, with no branch-specific exemption;
- the same holds under the **alternative** reference policy, reported alongside
  — a criterion that held only for the oracle would be a property of that
  trajectory rather than of the pair;
- equal entity-set cardinality and brief lengths within 15%;
- no family is silently omitted.

**The registered difficulty metric is the oracle**: `oracle_tool_calls` records
its count, and a test asserts that is what the field holds.

## Anti-padding guards

- No reference policy may repeat an identical read with no intervening state
  change. Two repeats are excused and only two: retrying a call that failed, and
  polling the approval thread for an asynchronous reply.
- Branch resolution may not reference the oracle, a trajectory, or a call count.
  The check parses the decision path with docstrings stripped, because those
  docstrings explain that it does not.

## A structural finding worth recording

W3's velocity signal originally created three extra charges and refunds, so
counterfactual instances were structurally *larger* than their siblings and
failed the entity-cardinality half of the criterion. Every world now carries the
same refund history and the signal varies **when** it happened — which is the
honest counterfactual anyway.
