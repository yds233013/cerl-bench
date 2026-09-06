# Corpus 2.0.0 and canonical split 1.2.0 — regeneration audit

```bash
uv run cerl freeze          # regenerate; byte-stable
uv run cerl splits          # partition sizes
uv run cerl splits --show-moves
```

Assertions read the committed files, not the generator:
`tests/scenarios/test_corpus_integrity.py`, `test_canonical_split.py`,
`test_holdout_integrity.py`.

## Criterion 42 — **PASS**

> *42. Lexicon shards pairwise disjoint across splits; no frozen training-split
> instance contains a held-out axis value.*

| Clause | Verdict | Evidence |
|---|---|---|
| Lexicon shards pairwise disjoint across splits | **PASS** | 0 shared values between any two partitions, over the committed files |
| No held-out value in the training split | **PASS** | 0 of 15 training scenarios, across every intervention axis |

Both clauses now hold literally, so the criterion is marked PASS.

## Why the corpus was regenerated, and why hashes changed

Corpus 1.x decided partitions **after** generation, from finished scenarios. By
the time anything knew which partition a scenario belonged to, its names had
already been chosen — so every file drew from the `core` pool and partitions
shared 15–17 entity names. No amount of relabelling could fix that: the names
were in the files.

Corpus 2.0.0 inverts the order. The partition is a function of the *plan* —
template, axes, seed — all of which exist before any entity does, so the
generator is handed the right shard and disjointness is a property of generation
rather than something to check for afterwards and be unable to repair.

**Hashes changed because the files changed.** Different names produce different
bytes. That is the point of the version bump rather than a cost of it: a
regenerated corpus that kept its hashes would mean nothing had actually been
fixed. The previous instruction to preserve hashes was explicitly superseded.

## Versions

| Artifact | 1.x | 2.0.0 |
|---|---|---|
| Corpus | `scenarios/frozen`, `scenarios/manifest.json` | `scenarios/v2/frozen`, `scenarios/v2/manifest.json` |
| Generator | `w2-1.0.0` | `w2-2.0.0` |
| Lexicon shards | `core`, `eval_a`, `eval_b`; all scenarios used `core` | `train`, `validation`, `evaluation`; shard version 2.0.0 |
| Split manifest | `scenarios/split_manifest.json` | `scenarios/v2/split_manifest.json` |
| Split rule | 1.0.0 → 1.1.0 → **1.2.0** | 1.2.0, unchanged by the regeneration |

**Corpus 1.x is untouched.** Its files, manifest and split manifest remain on
disk exactly as committed; `test_the_legacy_corpus_is_untouched` re-verifies
every 1.x sha256. `splits.partition_v1_0_0` remains computable, so 1.0.0 and
1.1.0 results stay interpretable.

## What is sharded

Every lexicon-backed value an agent can read, not only company names:

| Value | Train | Validation | Evaluation |
|---|---|---|---|
| Company names | 12 disjoint | 12 disjoint | 12 disjoint |
| Email domain | `tr-corp.example` | `va-corp.example` | `ev-corp.example` |
| Staff handles | `tr-*` | `va-*` | `ev-*` |
| Staff display names | 7 disjoint | 7 disjoint | 7 disjoint |

In 1.x, `DOMAIN_SHARDS` held **identical** tuples for all three shards — a shard
that sharded nothing — and staff were literals (`"Billing Manager"`) shared by
every scenario in the corpus. Either would have served as a memorisable
partition marker on its own.

The `.example` TLD is deliberately shared: it is what marks the data synthetic,
and it carries no partition information.

`lexicon.companies` no longer has a default shard. A default is precisely how
1.x came to draw everything from `core` — the parameter existed and no caller had
to pass it.

## Scenario counts by family and partition

| Family | train | validation | evaluation | total |
|---|---|---|---|---|
| `duplicate_billing_profile` | 5 | 13 | 18 | 36 |
| `duplicate_charge_approval` | 10 | 34 | 70 | 114 |
| `suspicious_refund_escalation` | 0 | 24 | 16 | 40 |
| **total** | **15** | **71** | **104** | **190** |

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
| **total** | **0** | **41** | **63** |

All 11 are namespaced `family/axis=value`. W1's `merge_approval` and W2's
`approval` share the value names `expired` and `missing_unobtainable`, so an
unnamespaced label would conflate two different workflows' counterfactuals.

## Lexicon overlap — measured over the committed files

Customer display names, customer emails, Slack handles and Slack display names.

| Pair | Shared values |
|---|---|
| evaluation ∩ train | **0** |
| evaluation ∩ validation | **0** |
| train ∩ validation | **0** |
| distinct values | train 46, validation 48, evaluation 50 |

Pool definitions and corpus contents also agree in the other direction: no file
contains a value belonging to another partition's pool
(`test_pool_definitions_and_corpus_contents_agree`).

## Sibling integrity

| Check | Result |
|---|---|
| Groups straddling a partition | **0** of 86 |
| CF scenarios whose ID sibling is elsewhere | **0** |
| Scenarios in more than one manifest group | **0** |
| Corpus scenarios missing from the manifest | **0** |
| Committed split manifest vs. fresh rebuild | identical |
| Scenario-id duplicates | **0** |
| Filename duplicates | **0** |
| Planned instances vs. files written | 190 = 190 |

## Criterion 41, mutations, determinism

| Check | Result |
|---|---|
| Criterion 41 matched pairs | **104 / 104** within ±1 (max Δ = 1) |
| Entity cardinality matched within pairs | pass |
| Brief length within ±15% within pairs | pass |
| W1 identity-evidence spread | 12 / 10 / 9, unchanged |
| Oracle clean 1.0 on every frozen scenario | 190 / 190 |
| Mutation suite | pass, no misclassifications |
| Determinism suite | pass |
| Re-freeze byte-stability | every file byte-identical |
| Cross-process / cross-version hashes | identical |

The difficulty spread is a property of the workflows, not of the names, so
renaming every entity left it untouched — which is the point of the C6 control.

## Training coverage: 5 of 10 branches

Not repaired, and the analysis the instruction asked for rather than an invented
fix.

### Why each missing branch is missing

| Branch | In-distribution values reaching it | Pure-ID groups | Why absent from training |
|---|---|---|---|
| `duplicate_billing_profile/distinct_entities` | `valid`, `missing_obtainable` | 3 | Its 3 pure-ID groups landed in validation/evaluation under the unchanged 60/15/25 hash |
| `duplicate_charge_approval/request_then_refund` | `missing_obtainable` | **0** | Every ID instance was frozen alongside a CF sibling, so every group is CF-bearing |
| `suspicious_refund_escalation/legitimate_refund` | `0` | **0** | Same: every `signal_count=0` instance has CF siblings at the same seed |
| `suspicious_refund_escalation/request_info` | **none** | 0 | Reachable only via `signal_count=1`, a registered holdout |
| `suspicious_refund_escalation/escalate_fraud` | **none** | 0 | Reachable only via `signal_count=2` or `3`, both registered holdouts |

Two distinct causes, and they call for different answers.

### Can principled training scenarios be added?

**For three branches, yes.** `distinct_entities`, `request_then_refund` and
`legitimate_refund` are all reachable from in-distribution values alone. They are
absent because of which instances were *frozen*, not because of what the axes
permit. Freezing additional **pure-ID** instances at fresh seeds — an ID value
with no CF sibling frozen at that seed — would create pure-ID groups eligible for
training, without touching a single held-out value and without changing any
hypothesis.

**For two branches, no.** `request_info` and `escalate_fraud` have no
in-distribution value at all. Giving them training scenarios would mean
un-registering `signal_count` 1, 2 or 3 as holdouts, which changes what the
counterfactual *is* for W3 and therefore changes H0 and C1–C6. That is a research
decision, not an implementation one, and it is not taken here.

### Alternatives, documented rather than chosen

| # | Option | Gains | Cost |
|---|---|---|---|
| **A** | Freeze additional pure-ID instances at fresh seeds for `request_then_refund`, `legitimate_refund`, and more `distinct_entities` | up to 8/10 training branches | Corpus grows; a new corpus version; no hypothesis change |
| **B** | Accept 5/10 and treat training as a small development pool | nothing to decide | The development pilot stays at 5 branches |
| **C** | Re-register W3 so one non-zero `signal_count` is in-distribution | 10/10 possible | **Changes the hypotheses.** W3's counterfactual becomes a different thing |
| **D** | Force one pure-ID group per branch into training | 8/10 | Biases the split. This is "moving scenarios to improve branch coverage", which is exactly what was forbidden |

**A is the only principled expansion**, and it is a corpus-authoring decision —
how many instances, at which seeds — left for a person. **C and D are recorded so
they are visibly rejected rather than silently unavailable.**

Nothing was invented, no axis value was added, and no scenario was moved to
improve coverage.

## Pilot under the regenerated corpus

| Config | Episodes | Branches | Estimated | Worst case | Cap |
|---|---|---|---|---|---|
| 1/branch | 5 | 5/10 | $1.08 | $12.84 | $13 |
| 2/branch | 9 | 5/10 | $1.91 | $23.13 | $24 |

### Selected scenarios — 5-episode development pilot

| # | scenario_id | family/branch |
|---|---|---|
| 1 | `dup_profile_reconciliation__disp-none__id-name_only_similarity__appr-valid__pp-none__tr-stable__s101` | duplicate_billing_profile/escalate_ambiguous |
| 2 | `dup_profile_reconciliation__disp-none__id-strong_match__appr-valid__pp-none__tr-search_flaky__s17` | duplicate_billing_profile/merge_sanctioned |
| 3 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-short__nd-absent__pp-none__t-10000__tr-stable__s101` | duplicate_charge_approval/escalate_unapproved |
| 4 | `dup_charge_threshold__amt-below_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-search_flaky__s101` | duplicate_charge_approval/refund_below_threshold |
| 5 | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-standard__nd-absent__pp-none__t-10000__tr-refund_timeout_once__s17` | duplicate_charge_approval/refund_now |

All five are in the canonical training partition, carry no registered held-out
value, and draw only from the `train` lexicon shard.

## Unchanged

- **W1's research scope.** The matched-pair set (15 pairs on `merge_approval`)
  and the separately-reported `identity_evidence` challenge set remain distinct
  and both survive the regeneration; see `docs/w1-scope.md`.
- **Criterion 41.** 104 matched pairs, all within ±1.
- **Evaluation coverage.** Validation and evaluation each cover 10/10 branches.
