# Two-rollout engineering smoke test — protocol v2 backward + optimizer path

Evidence index, with every artefact linked: [`INDEX.md`](INDEX.md).

## What this is, and what it is not

This is an **engineering smoke test**: does `turn_backward` plus one optimizer
step run to completion on real weights, real recorded token spans, and a real
reward-derived advantage, inside a 20-minute budget?

It is **not**:

* **Not the four-rollout update.** That group is rollouts 0–3. This uses two of
  them, so it is a *different* group with different advantages and a different
  gradient. The four-rollout attempt remains unfinished.
* **Not a research performance result.** The two rollouts were selected **after
  inspecting the previous run**, by the rule "the first two in original order".
  The reason for cutting the group at all is that the full four-rollout backward
  pass — 36 turns over 80,279 tokens — did not fit in 20 minutes. Choosing a
  subset after seeing the cost is a legitimate engineering decision and an
  illegitimate experimental one.

  These are **not the two cheapest** of the four. Rollout 2 is cheaper than
  rollout 1, so the cheapest pair is {0, 2} at 9,763 sequence tokens; the pair
  actually used is {0, 1} at 15,409. What first-two-in-original-order does is
  exclude **rollout 3**, which alone is 24 of the 36 turns and 74.7% of the
  group's work. (The run record `smoke.json` carries the superseded "cheapest"
  wording in its `selection.why` field; it is left unchanged, and the erratum is
  in [`INDEX.md`](INDEX.md).)
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
| share of the four-rollout work | 6.0% | 13.2% | 19.2% |
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
| **total elapsed (child)** | **768.7 s** of the 1200 s budget — see *Timing* |

`lora_A` receiving exactly zero gradient is expected, not a defect: the gradient
w.r.t. `A` is proportional to `B`, and `B` is zero-initialised, so on the *first*
step only `B` can move. L∞ equal to the learning rate is likewise AdamW's
first-step behaviour (the update is `lr · sign`).

## Timing — child and supervisor are different durations

Two clocks appear in the evidence and they do not agree. They are measuring
different things.

| Measurement | Value | Source |
|---|---|---|
| Child process, self-timed, launch to final save | **768.7 s** | `smoke.json` → `total_seconds` (a `time.monotonic()` span inside the process) |
| — of which model load | 6.8 s | `smoke.json` → `config.load_seconds` |
| — of which backward | 760.1 s | `smoke.json` → `backward.seconds` |
| Child process, independent bracket | **769 s** | file mtimes: `stderr.log` created 21:11:27 at launch; `smoke.json` and `stdout.log` last written 21:24:16 |
| Supervisor shell, `start` to `end` echo | **1200 s** | 21:11:27 → 21:31:27 in the run log |

**The child ran for ~769 s.** The two independent measurements of it — the
process's own monotonic timer and the output-file mtime bracket — agree to
within a second.

**The supervisor's 1200 s is not a measurement of the child.** It equals the
watchdog budget exactly, to the second. The watchdog did not fire: `stderr.log`
is 0 bytes, so its termination notice was never written, and the child exited 0.
The consistent explanation is that `kill "$WD"` ends the watchdog subshell but
not the `sleep 1200` it spawned; that surviving `sleep` inherited the pipeline's
stdout, so the reader saw no end-of-file until the budget elapsed, roughly 431 s
after the child was already finished. The exact equality with the budget is the
evidence for this; there is **no independent timestamp** of the shell's own
return, so the ~431 s figure is a difference between the two brackets above and
not a separately measured quantity.

Nothing about the result depends on which clock is used: the child finished on
its own, well inside the budget, and was never signalled.

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
policy got better, and nothing about the four-rollout group. **Post-update task
performance remains unmeasured**, and measuring it would require inference this
milestone does not include.
