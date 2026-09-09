# Evidence index — corrected v2 training-integration milestone

One reward-driven optimizer update completed, from two previously recorded
training rollouts. Engineering demonstration; **post-update task performance is
unmeasured.** Read [`REPORT.md`](REPORT.md) for what that does and does not mean.

## Configuration

| | |
|---|---|
| Base model | cached, pinned Qwen3-0.6B (local, no download) |
| Adapter | fresh rank-8 LoRA, seed **20260908**, all `lora_B` verified zero before the step |
| Objective | `cerl_rl.grpo.turn_backward`, β_KL = 0 |
| Optimizer | AdamW, lr 1e-5, weight_decay 0 |
| Device | MPS |
| Full record | [`smoke.json`](smoke.json) → `config`, `group` |

## Source-rollout provenance

Rollouts **0 and 1** of the four recorded by the trial at commit `6f361d90`.
No new generation was performed.

| | |
|---|---|
| Recorded turns, token spans, actions, outcomes | [`../rl-v2-trial2/turns.jsonl`](../rl-v2-trial2/turns.jsonl) |
| Replay verification of that trial | [`../rl-v2-trial2/replay.json`](../rl-v2-trial2/replay.json) |
| Trial report | [`../rl-v2-trial2/REPORT.md`](../rl-v2-trial2/REPORT.md) |
| Preconditions re-checked here, on **all four** rollouts | [`smoke.json`](smoke.json) → `preconditions_on_all_four` (`passed: true`) |

## Progress log

[`backward_progress.jsonl`](backward_progress.jsonl) — one fsynced line per
completed turn backward (9 lines), carrying episode, step index, sequence
length, generated tokens, running loss, running tokens and elapsed seconds. A
timeout would have named exactly how far computation reached.

## Checkpoint

| | |
|---|---|
| File | [`adapter/adapter_model.safetensors`](adapter/adapter_model.safetensors) |
| sha256 | `87b156f884f0eea44581c5917ba50b3a8c8dbd61f6ac90a7194475ba5047ad98` |
| Size | 9,204,512 bytes |
| Verify | `shasum -a 256 evidence/rl-v2-smoke-2rollout/adapter/adapter_model.safetensors` |

## Reproduction

```bash
# under the external watchdog (as run), 1200 s budget covering model loading
./scripts/run_v2_smoke2.sh 1200

# or directly
PYTHONPATH=src .venv-rl/bin/python scripts/v2_two_rollout_smoke.py
```

Script: [`../../scripts/v2_two_rollout_smoke.py`](../../scripts/v2_two_rollout_smoke.py).
Watchdog: [`../../scripts/run_v2_smoke2.sh`](../../scripts/run_v2_smoke2.sh).
Requires the cached base model and the isolated `.venv-rl` training environment;
neither is part of the ordinary install or of offline CI.

## Erratum

`smoke.json` was written by the script as it stood at run time, and its
`selection.why` field says the two rollouts "are the cheapest of the four". That
is wrong: rollout 2 (4,923 sequence tokens) is cheaper than rollout 1 (10,569),
so the cheapest pair is {0, 2} at 9,763 tokens, not the {0, 1} pair used here at
15,409. The selection rule actually applied was **the first two in original
order**, which excludes rollout 3 and its 74.7% share of the work.

The run record is left byte-unchanged, because it is the record of what the run
wrote. The script's wording is corrected, so a re-run records the accurate
rationale — which means `smoke.json` and a fresh run's `selection.why` will
differ in that field, and only in that field.

## Not in this milestone

The original **four-rollout** update (rollouts 0–3) never completed — its
backward pass exceeded the 20-minute budget. That attempt is preserved
unchanged in [`../rl-v2-recovery/`](../rl-v2-recovery/) and is not superseded by
this run.
