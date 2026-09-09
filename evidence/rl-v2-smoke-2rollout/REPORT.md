# Two-rollout engineering smoke test — protocol v2 backward + optimizer path

## What this is, and what it is not

This is an **engineering smoke test**: does `turn_backward` plus one optimizer
step run to completion on real weights, real recorded token spans, and a real
reward-derived advantage, inside a 20-minute budget?

It is **not**:

* **Not the four-rollout update.** That group is rollouts 0–3. This uses two of
  them, so it is a *different* group with different advantages and a different
  gradient. The four-rollout attempt remains unfinished.
* **Not a research performance result.** The two rollouts were selected **after
  inspecting the previous run** — they are the two cheapest of the four, chosen
  because the full group's 36 backward passes over 80,279 tokens did not fit in
  20 minutes. Picking the cheap half after seeing the cost is a legitimate
  engineering decision and an illegitimate experimental one.
* **Not a measurement of task performance.** No evaluation was run before or
  after the update. Task performance after this step is **explicitly unmeasured**.

The four-rollout evidence in `evidence/rl-v2-trial2/` is unchanged. No new
generation was performed; every token came from the recorded trial.

## Inputs

Preconditions were re-checked on **all four** recorded rollouts before selecting
two (`preconditions_on_all_four` in `smoke.json`, `passed: true`): contiguous
step indices, exact prompt/generated token spans present on every turn, original
termination reached, and replay reproducing every outcome and reward.

| | rollout 0 | rollout 1 | group |
|---|---|---|---|
| turns | 3 | 6 | 9 |
| generated tokens | 64 | 218 | 282 |
| sequence tokens (fwd+bwd) | 4,840 | 10,569 | **15,409** |
| replay-derived reward | 0.366667 | 0.066667 | |
| **recomputed advantage** | **+1.0** | **−1.0** | |
| termination | declared:escalate | declared:finish | |

The advantages are recomputed for *this* group. With two members the
group-relative advantage is necessarily ±1, which is itself a reason this
configuration is a smoke test and not an experiment.

## Configuration

Cached pinned Qwen3-0.6B; fresh rank-8 LoRA seeded with the recovery's recorded
seed **20260908**; `cerl_rl.grpo.turn_backward`; AdamW, lr 1e-5, weight_decay 0,
β_KL = 0. Model load: 6.8 s. Device: MPS.

Pre-update policy equivalence: **all `lora_B` tensors exactly zero**, so the
adapter contributed nothing and the starting policy was the base model.

## Result

| | |
|---|---|
| status | **completed — one optimizer step** |
| turns backwarded | 9 / 9 |
| scored tokens | 282 |
| final loss | −0.0592559 |
| **gradient norm** | **1.8240920** (finite, 224 tensors with grad) |
| optimizer step | taken (gradient neither zero nor non-finite) |
| **parameter change, L1** | **11.4658** |
| **parameter change, L∞** | **9.99999883788405e-06** |
| tensors changed | 112 of 224 |
| — `lora_B` changed | 112 of 112 |
| — `lora_A` changed | **0 of 112** |
| **checkpoint sha256** | `87b156f884f0eea44581c5917ba50b3a8c8dbd61f6ac90a7194475ba5047ad98` |
| checkpoint size | 9,204,512 bytes |
| **backward elapsed** | 760.1 s |
| **total elapsed** | **768.7 s** of the 1200 s budget |

`lora_A` receiving exactly zero gradient is expected, not a defect: the gradient
w.r.t. `A` is proportional to `B`, and `B` is zero-initialised, so on the *first*
step only `B` can move. L∞ equal to the learning rate is likewise AdamW's
first-step behaviour (the update is `lr · sign`).

## Per-turn progress

`backward_progress.jsonl` was fsynced after each completed turn's backward pass,
so a timeout would have identified exactly how far computation reached. All nine
lines are present. Wall-clock cost per turn grew with sequence length:

```
ep0 t0  1,543 tok   39 s      ep1 t0  1,556 tok  151 s
ep0 t1  1,613 tok   70 s      ep1 t1  1,627 tok  183 s
ep0 t2  1,684 tok  118 s      ep1 t2  1,695 tok  235 s
                              ep1 t3  1,816 tok  356 s
                              ep1 t4  1,900 tok  551 s
                              ep1 t5  1,975 tok  767 s
```

The marginal cost per turn rises steeply — the last turn alone took ~215 s. At
this rate the full four-rollout group (36 turns, 80,279 tokens, sequences to
3,449) is far beyond 20 minutes, which is consistent with the earlier timeout
and is the constraint any future full-group run has to solve.

## Conclusion

The corrected v2 training machinery — recorded-provenance episodes → replay-
derived rewards → group-relative advantages → per-turn backward → optimizer step
→ atomic checkpoint — **runs end to end on real weights.** The adapter moved.

That is a statement about the machinery. It says nothing about whether the
policy got better, and nothing about the four-rollout group. Task performance
after this update remains unmeasured.
