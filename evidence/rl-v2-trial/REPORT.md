# Bounded protocol-v2 training trial — failed on an MPS allocation limit

**No rollout completed, no optimizer update occurred, and no agent completed any
task.** The trial aborted during its first generation. Nothing about learning is
established, in either direction.

## Configuration, recorded before the run

`config.json`, written before any model was loaded.

| | |
|---|---|
| Scenario | `dup_charge_threshold__amt-above_threshold__appr-valid__ttl-short__nd-absent__pp-none__t-10000__tr-stable__s101` |
| | first in protocol order, branch `escalate_unapproved`, required decision **escalate**, canonical **train** partition |
| Model | `Qwen/Qwen3-0.6B` @ `c1899de…47ca`, cached; **fresh** rank-8 LoRA (the historical adapter was not loaded) |
| Path | protocol v2 — `rollout_v2` + `turn_backward`, complete tool contract |
| Limits | 24 actions, 8,192 prompt tokens, 160 output tokens |
| Sampling | temperature 1.0 with `top_k=0, top_p=1.0, min_p=0, typical_p=1, repetition_penalty=1` |
| Trial | 1 group × 4 rollouts, **at most one** optimizer update, lr 1e−5 |
| Validation | not run; **no validation scenario was loaded** (`--training-only`) |
| Budget | 20 min external watchdog, 18 min in-process deadline, both covering model loading |

## What happened

```
MPSNDArray.mm:788: failed assertion
`[MPSTemporaryNDArray initWithDevice:descriptor:] Error: total bytes of NDArray > 2**32'
```

A single Metal allocation exceeded **4.295 GB (2³²)**. That is a hard assertion,
not a Python exception: it aborts the process, so the pilot's own handler never
ran and could not label the record. Everything in `pilot_run.json` above the
`outcome` key was written by the pilot before the crash; `outcome` was added
afterwards by inspection and says so.

It happened inside the **first** `generate` of rollout 0 — no turn completed, so
`turns.jsonl` was never created and no checkpoint was written. The 18-minute
in-process deadline and the 20-minute watchdog were never reached; the run died
at roughly 30 seconds.

## Answers to the four questions

1. **Per-rollout reward, task completion, safety** — none. Zero rollouts
   completed.
2. **Did an update occur?** No. Zero optimizer updates, no gradient norm, no
   parameter change, no checkpoint. This is *not* the "identical rewards or zero
   gradient" skip case — the trial never reached the point of having rewards.
3. **Replay, elapsed time, interruption** — `verify_run` reports
   `status: empty, ok: false, episodes_replayed: 0`: there is genuinely nothing
   to replay, which is the correct report rather than a vacuous pass. Elapsed
   ≈30 s of a 20-minute budget. The interruption was an abort, not a timeout.
4. **Did an agent complete the task?** **No.** No action was ever executed
   against the environment.

## What is and is not established

The failure is **specific to generation**, not to the model or the prompt: the
logit diagnostic ran four successful forward passes on *this same* 1,524-token
prompt on MPS in bfloat16 two commits ago. Sizes at that length are nowhere near
the limit — eager attention weights 0.15 GB, full-sequence logits 0.93 GB in
fp32. So the offending allocation has **not been identified**, and I did not
bisect it, because the instruction was to stop on out-of-memory and after this
single trial. Raising any memory limit was not attempted.

One measured fact worth carrying forward: at the v2 cap of **8,192 prompt
tokens**, eager attention weights alone are `16 × 8192² × 4 B` = **4.29 GB**,
which sits essentially exactly on the 2³² ceiling. `protocol_v2` already flagged
this as unverified — *"whether an 8,192-token forward pass fits comfortably in
this machine's memory during training is not established here"*. That warning
now has a failure next to it, though the crash occurred at 1,524 tokens, so the
8,192 figure is **not** the demonstrated cause.

## What was preserved

- The historical pilot (`ef6942e`), its adapter (sha256 `96391951…41ff4`,
  re-verified) and the logit diagnostic are **untouched**; the v1 run still
  replays all 58 episodes.
- The trial wrote only to `evidence/rl-v2-trial/`.
- No download, no paid call, no validation scenario, nothing pushed.

## Suggested next step — not run here

Isolate the allocation with a single bounded `generate` call on this exact
prompt, outside the training loop, bisecting `max_new_tokens` and sampling
arguments. That is one short model operation, not a training run, and it is the
cheapest way to tell an MPS/transformers generation issue from a protocol-v2
sizing problem. Until it is understood, raising or lowering limits would be
guessing.
