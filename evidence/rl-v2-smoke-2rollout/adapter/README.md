---
base_model: Qwen/Qwen3-0.6B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-0.6B
- lora
- transformers
---

# CERL-Bench — two-rollout smoke-test LoRA adapter

**This adapter is evidence, not a model release. Do not use it for anything.**

It is the checkpoint produced by a single engineering smoke test: one optimizer
step over two previously recorded training rollouts, run to prove that the
CERL-Bench v2 training path executes end to end on real weights.

**Post-update task performance is unmeasured.** No evaluation was run before or
after the step. There is no evidence that this adapter is better than the base
model at anything, and no claim that it is.

## What produced it

| | |
|---|---|
| Base model | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0) |
| Adapter | rank-8 LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`; α=16, dropout 0 |
| Trainable parameters | 2,293,760 |
| Initialisation seed | 20260908; all `lora_B` verified exactly zero before the step |
| Objective | `cerl_rl.grpo.turn_backward`, group-relative advantages, β_KL = 0 |
| Optimizer | AdamW, lr 1e-5, weight_decay 0 — **one step** |
| Training data | 2 recorded rollouts, 9 turns, 282 scored generated tokens |
| Gradient norm | 1.8240920 |
| Result | 112 `lora_B` tensors moved; `lora_A` unmoved, as expected on the first step |
| sha256 | `87b156f884f0eea44581c5917ba50b3a8c8dbd61f6ac90a7194475ba5047ad98` |

The rollouts were **selected after inspecting earlier results** — rollouts 0 and
1 in their original order — so the group is post-hoc and is not an experimental
sample. All data is synthetic; the environment contains no real customer,
company or personal data.

## License

Apache-2.0, matching both this repository and the Apache-2.0 base model it
adapts.

## Full provenance

`../INDEX.md` in this repository links the configuration, the source rollouts,
the per-turn progress log, the checksum and the reproduction command.
`../REPORT.md` states what the run does and does not establish.

*(This card replaces the empty template PEFT writes by default. The weights and
the run record `../smoke.json` are unchanged.)*
