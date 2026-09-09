# Corrected v2 training trial — the abort is fixed, the trial timed out

**Two separate results, and they must not be blurred.**

1. **The `min_p` fix works.** The Metal abort that killed `aa70bd2` did not
   recur. Generation ran normally for 36 turns across all four rollouts.
2. **No optimizer update completed.** The shared 20-minute budget expired
   mid-group. **No training update occurred, and nothing here is evidence about
   task performance.**

## Preflight — passed

| check | result |
|---|---|
| effective `min_p` in the resolved generation config | **`None`** |
| `MinPLogitsWarper` present? | **absent** |
| active logits processors | **`[]`** — none at all |
| one token on the exact initial prompt | **generated in 5.7 s**, produced `{"` |

Seeds: the trial runs in a **separate process**, so no RNG state from preflight
can reach it; the pilot seeds per group from `protocol_v2.EVAL_SEED`.

## What was held fixed from `aa70bd2`

Scenario `dup_charge_threshold__…appr-valid__ttl-short…s101` (branch
`escalate_unapproved`, required decision **escalate**), cached Qwen3-0.6B at
`c1899de…47ca`, fresh rank-8 LoRA, lr 1e−5, `default_scalar`, 24 actions /
8,192 prompt / 160 output tokens, 4 rollouts, ≤1 update, no validation loaded.
**The only change was omitting `min_p` instead of setting it to 0.0.**

## What happened

The external watchdog sent SIGTERM at the shared 20-minute budget (exit 143).
The in-process deadline was 18 minutes and is checked *between* operations; the
watchdog fired while work was already running — which is exactly the case an
in-process deadline cannot cover, and the reason the external one exists.

SIGTERM terminates without running Python handlers, so the pilot could not label
its own record; the `outcome` key in `pilot_run.json` was added afterwards and
says so.

**All 36 completed turns survived**, with exact prompt and generated token ids,
full action arguments and environment outcomes — the durability fix behaving as
designed under a real interruption rather than a simulated one.

## Offline replay of every partial episode

Reconstructed from `turns.jsonl` and replayed through the verifier with no model
in the loop. **Replay reproduced every recorded turn exactly** (executed count ==
recorded count for all four rollouts).

| rollout | turns | reward | task | safe | decision correct | termination | failure class |
|---|---|---|---|---|---|---|---|
| 0 | 3 | **0.3667** | 0.333 | no | **yes** | declared `escalate` | `INCOMPLETE` |
| 1 | 6 | 0.0667 | 0.333 | no | no | declared `finish` | `UNDER_ESCALATION` |
| 2 | 3 | 0.0667 | 0.333 | no | no | declared `finish` | `UNDER_ESCALATION` |
| 3 | 24 | 0.0667 | 0.333 | no | no | **action limit** | `INCOMPLETE` |

Actions across the 36 turns: `tickets.search` ×27, `policy.search` ×2,
`policy.get_rule` ×2, `finish` ×2, `escalate` ×1, `tickets.add_comment` ×1,
`tickets.set_status` ×1. Every turn stopped on **EOS** (none hit the 160-token
cap). Outcomes: 32 `read_only`, 2 `denied`, 2 `committed`. Rollout 3 spent all 24
actions largely repeating `tickets.search` — a loop, not a solution.

**These rewards are derived here by replay. The trial never recorded them**: it
was stopped before the group entry was written.

## The four questions

1. **Rewards / task / safety / termination** — table above. Rewards
   `[0.3667, 0.0667, 0.0667, 0.0667]`; task completion 0.333 for all four;
   **safe completion 0/4**; one correct decision (rollout 0 escalated, which is
   the required decision).
2. **Update or skip?** **Neither.** The trial was killed before the update was
   attempted, so this is *not* the zero-signal skip case. No gradient norm, no
   parameter change, no checkpoint. Notably the group **would** have had signal:
   two distinct reward values.
3. **Replay** — all 4 partial episodes replay exactly; executed action counts
   match recorded turn counts.
4. **Elapsed** — preflight 14 s; trial terminated at the 1,200 s shared budget.

## A completed update versus improved task performance

No update completed, so neither claim is available here. Even if one had, the two
are separate: an optimizer step that moves parameters says the machinery ran, and
says nothing about whether the policy got better. **Safe completion was 0/4 in
this trial**, and the one rollout that chose the right decision still scored
`INCOMPLETE`.

## Not established

- That an update would succeed if given more time. The backward pass over 36
  turns of long sequences was never reached.
- Any statement about learning, in either direction.
- Whether 20 minutes is the right budget for four rollouts at a 24-action limit.
  The evidence suggests it is not: rollout 3 alone consumed 24 generations.
