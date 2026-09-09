# Recovery from recorded rollouts — timed out in the backward pass

**Recovery of one update from the rollouts recorded by the trial at
`6f361d90b593c1e749a43dbd8c026135bed818c0`.** No new generation: every token used
here was produced by that trial.

**No optimizer update occurred.** The 20-minute external budget expired inside
`turn_backward`. Everything before that point passed and is recorded.

## Preconditions — all passed

Checked offline, no model. Any failure would have stopped the run rather than
training on a shortened group.

| rollout | turns | termination | reward | task | safe | decision correct | generated tokens |
|---|---|---|---|---|---|---|---|
| 0 | 3 | declared `escalate` | **0.3667** | 0.333 | no | **yes** | 64 |
| 1 | 6 | declared `finish` | 0.0667 | 0.333 | no | no | 218 |
| 2 | 3 | declared `finish` | 0.0667 | 0.333 | no | no | 108 |
| 3 | 24 | **action limit** | 0.0667 | 0.333 | no | no | 784 |

- All four reached their **original** termination — three terminal declarations,
  one at the 24-action limit. None was cut short.
- Every turn carries its exact prompt tokens, generated tokens, full action and
  environment outcome; step indices are contiguous from zero.
- **Replay reproduces the outcomes and the rewards**, and agrees with the trial's
  own `replay.json` to 1e−9.

## Policy equivalence — verified, with one honest limitation

| | |
|---|---|
| `lora_B` tensors | 112, **all exactly zero** |
| `lora_A` tensors | 112 |
| adapter-enabled vs disabled logits | **bit-identical** on a recorded 1,524-token prompt |

LoRA computes `base + scaling · B @ A`. With every `B` exactly zero the adapter
contributes nothing, so the pre-update policy **is** the pinned base model. That
was verified by comparing logits, not assumed.

**Limitation, stated plainly.** The original trial recorded no LoRA
initialisation seed, so its exact `A` matrices cannot be recreated. That does not
affect the pre-update *policy* — identical for any `A` while `B` is zero — but it
does affect the *update*, because `dL/dB` depends on `A`. Any parameter change
computed here would not be bit-identical to the one that trial would have
produced. This recovery is seeded (`20260908`) and reproducible going forward.

## The update — not reached

Reached, in order: preconditions → policy equivalence → group rebuilt through the
v2 structures → advantages computed → `turn_backward` entered.

```
rewards      [0.3667, 0.0667, 0.0667, 0.0667]   2 distinct values
advantages   [1.7321, -0.5774, -0.5774, -0.5774]
group        36 turns, 1,174 generated tokens
```

Then the watchdog fired.

| | |
|---|---|
| backward completed | **no** |
| gradient norm | not reached |
| update occurred | **no** |
| parameter changes | none |
| checkpoint | none written |
| elapsed | model load 7.8 s; terminated at the 1,200 s budget |

## Why it did not fit

`turn_backward` runs **one forward and one backward per turn**, over
`prompt + generated`:

| rollout | turns | sequence length | tokens |
|---|---|---|---|
| 0 | 3 | 1,543–1,684 | 4,840 |
| 1 | 6 | 1,556–1,975 | 10,569 |
| 2 | 3 | 1,556–1,730 | 4,923 |
| 3 | 24 | 1,556–**3,449** | **59,947** |
| | **36** | | **80,279** |

Rollout 3 alone is three quarters of the work: it ran to the 24-action limit, and
because each turn re-includes the whole conversation, its sequences grow to
3,449 tokens. 36 passes over 80,279 tokens did not fit in 20 minutes here.

## What this establishes

Nothing about task performance, and not yet that the machinery completes a full
update on real weights either. What it does establish:

- the recorded rollouts are **complete and faithful** — replay reproduces every
  outcome and reward;
- the recovery path rebuilds them through the real v2 structures and computes the
  correct group-relative advantages;
- the pre-update policy is **provably** the pinned base model.

**A completed update would establish that the corrected training machinery runs.
It would not establish improved task performance** — and there is no completed
update here in any case. Safe completion across the four recorded rollouts is
**0/4**.

## Not established

- Whether the backward completes given more time. It was entered, not finished.
- Any gradient norm or parameter change for this group.
- Whether the earlier zero-gradient risk applies here. Advantages are non-zero
  and the group has real variation, but the gradient was never computed.
