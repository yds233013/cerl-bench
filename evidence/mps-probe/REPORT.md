# The v2 trial's Metal abort, isolated

**Cause found: `min_p=0.0`.** Passing zero to mean "no min-p filtering" does not
disable the min-p warper — it *builds* it, and the warper's whole-vocabulary
operations abort the process on MPS.

Four probes, 10-minute shared budget, each in its own child process. No training,
no episode, no download, no dependency change, no memory limit raised.

## Probes and results

| # | probe | result |
|---|---|---|
| 1 | sampling ops on a synthetic real-vocabulary tensor, **no weights loaded** | **passed** — `softmax`, `multinomial`, `argmax`, `sort` all fine at `(1, 151936)` fp32 on MPS |
| 2 | one token, exact v2 sampling settings | **ABORTED**, exit 134, at `v2_sampling_generate_begin`, prompt 1,524 tokens, `max_new_tokens=1` |
| 3 | one token, greedy — only the sampling mode changed | **passed**, 11.8 s |
| 4 | targeted bisect of the sampling arguments, one at a time | **ABORTED at exactly one case** |

Probe 4's marker trail, each entry fsynced and MPS-synchronised so the last one
is meaningful:

```
 6.96s  case_begin  a_sample_defaults      {"do_sample": true}                        -> done
12.93s  case_begin  b_sample_temperature   + temperature 1.0                          -> done
18.46s  case_begin  c_plus_top_k_0         + top_k 0                                  -> done
23.88s  case_begin  d_plus_top_p_1         + top_p 1.0                                -> done
29.41s  case_begin  e_plus_min_p_0         + min_p 0.0                                -> ABORT
```

Everything through `top_p=1.0` completed. Adding **`min_p=0.0`** aborted with

```
MPSNDArray.mm:788 failed assertion `[MPSTemporaryNDArray initWithDevice:descriptor:]
Error: total bytes of NDArray > 2**32'
```

## Why

Read from the installed `transformers==4.57.1`, not inferred:

```python
if generation_config.min_p is not None:
    MinPLogitsWarper(min_p=..., min_tokens_to_keep=...)
```

**`0.0 is not None`.** Every other neutralised setting is guarded by a *value*
test — `top_k != 0`, `top_p < 1.0`, `typical_p < 1.0`, `repetition_penalty != 1.0`,
`no_repeat_ngram_size > 0`, `renormalize_logits is True` — so passing the neutral
value genuinely disables it. `min_p` was the only bare `is not None` among the
eight, which is why neutralising it by value was the one that backfired.

The constructed warper then runs `argsort`, `gather` and `scatter` across the
full 151,936-token vocabulary. Probe 1 showed `sort` alone is fine, so the
offending allocation is inside one of the remaining ops — **which specific one is
not established.** Pinning that down needs a fifth probe, and the budget was four.

## Fix, kept separate from the historical trial

`protocol_v2.SAMPLING_NEUTRALISED` no longer contains `min_p` at all. Omitting
the key leaves it `None`, so no warper is constructed; the intended semantics —
no min-p filtering — are unchanged.

Two tests guard it: one asserts the key is absent with the reason, and a second
(model-marked) reads the installed `_get_logits_processor` source and asserts
that **every** remaining neutralised key is guarded by a value test, so another
release cannot quietly reintroduce the same trap.

**The historical trial record at `aa70bd2` is untouched.** It documents what it
actually ran, `min_p: 0.0` included.

## What is not established

- The exact failing operation inside the min-p warper.
- That the corrected configuration now generates successfully. That needs a model
  run, which is outside this diagnostic. The fix is verified statically — the
  warper is not constructed — and by the probe-4 case immediately before it
  (`top_p=1.0` without `min_p`) completing.
- Whether the separately noted 8,192-token prompt cap poses its own MPS
  allocation risk. Unrelated to this abort, still unverified.
