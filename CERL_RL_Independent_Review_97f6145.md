# Independent review: CERL-Bench local RL pilot

Reviewed candidate: `97f61458fa9c1ba8a288eaa89bd1b4c4997c25bc`  
Historical pilot: `ef6942ef21a3a7b1e6e24805dd044f6f2e0ed8af`  
Review date: 2026-09-08

## Decision

Preserve the pilot as a recorded negative development experiment. Repair the training wrapper before another model run. The saved results support reward-weighted parameter updates and unchanged validation behavior, but the implementation does not satisfy its claimed exact on-policy, generated-token-only training contract.

The most consequential findings are in token provenance and conditioning, the advertised tool contract, and sampling. They are separate from the already-known action/context limits. Increasing the learning rate would not correct these defects.

No pretrained model weights were downloaded, no pretrained model inference or training was performed, and no paid service was used in this review. The pinned tokenizer and configuration files were downloaded for tokenization checks. Gradient tests used small synthetic CPU networks.

## What was checked

- Archive SHA-256 matched `66481069599888f8ef86d92578707c8583373b91e44134c9e74a7ace60244fa5`.
- All 24 entries in the packaged `SHA256SUMS` verified.
- The adapter checksum matched the recorded pilot; all 112 saved `lora_B` tensors were nonzero.
- Independently replayed the 5 baseline, 48 training, and 5 after-evaluation action sequences. Trace hashes and recorded reward/task/safety results reproduced. Baseline and after terminal state hashes reproduced too.
- The 48 training episodes had no correct final state; 11 had the correct decision. Their reward variation came from the decision term and partial task credit.
- Before/after validation action sequences were identical, with 0/5 safe task completion in both phases.
- Inspected the code path: after-evaluation receives the same model object updated by the optimizer. Disk-reload output equivalence and the adapter's effect on logits remain unverified.

The package omits the simulator source and frozen scenarios. For these offline checks, it was paired with the previously reviewed `3d7fc20` simulator and canonical corpus. The packaged split manifest was used. This is not an independent checkout of the unpublished full `97f6145` repository.

## Test execution

Environment: Linux CPU, Python 3.12, PyTorch `2.8.0+cpu`, Transformers `4.57.1`, PEFT `0.17.1`. No MPS device is available here.

| Execution | Result |
|---|---|
| Packaged RL tests, unmodified | **60 passed, 7 failed, 5 skipped** |
| Same tests, with only `torch.mps.empty_cache` replaced by a no-op in the diagnostic process | **67 passed, 5 skipped** |

Every original failure was the unconditional MPS cache call in `group_backward`, not a loss/gradient assertion. The diagnostic replacement did not change the arithmetic or the source files. The five skipped tests require real model weights. One pytest warning concerned the model marker in the partial package's test configuration.

The passing gradient comparisons show that the optimized loss agrees with the supplied reference **given a mask and token sequence**. They do not establish that the real rollout constructed the correct mask or sequence. That distinction is where the principal defect lies.

## R1 — P1: Loss token provenance and conditioning are incorrect

Source: `src/cerl_rl/rollout.py`, `rollout`, `_generated_mask`, `_messages`; `src/cerl_rl/grpo.py`, `group_backward`.

There are two independently reproduced problems.

### R1a: A text match can select a system or observation span

The implementation decodes generated tokens to text, re-renders the completed conversation, and searches the entire resulting token sequence for each completion. The search begins at position zero. It does not restrict matches to the assistant turn that produced the text.

Using the actual pinned Qwen tokenizer, a system message containing a JSON answer after a newline, followed by an assistant producing that same answer, caused **all 32 selected tokens to come from the system message**. A one-token completion `You` likewise selected the `You` in `You are a support agent.`

The exact production system example with its preceding space did not collide in that particular test because of tokenization boundaries. The reproduced defect is not a claim that every historical completion collided, or that this particular example necessarily did. It demonstrates that the general mask construction is unsound.

If a completion cannot be matched, `_generated_mask` silently skips it. Special tokens, including a generated end-of-sequence token, are removed before reconstruction. The saved pilot rows contain actions but do not retain the raw per-turn input/generated token IDs needed to audit every historical training span.

### R1b: Re-rendering changes the context used to score earlier actions

Using the pinned tokenizer and the production message-building functions, the first action of a two-turn conversation was generated after this suffix:

```text
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The completed conversation used for backward scored that same earlier action after:

```text
<|im_start|>assistant\n
```

Qwen's template omits the empty thinking block on earlier assistant messages when the full conversation is re-rendered. The conditional context therefore differs. Even with matching weights, this loss is not evaluating the exact conditional probabilities from generation. Turning off top-k alone cannot fix this.

### Required correction

- Capture exact prompt token IDs and generated token IDs at each `generate` call, before decoding.
- Preserve the exact context under which each generated token was sampled. Per-turn records are appropriate when the next prompt re-renders earlier messages.
- Construct loss spans from recorded positions, never substring search.
- In each per-turn loss, earlier assistant turns that are now context must not be scored again. Preserve the intended group/episode weighting explicitly.
- Record generated stopping tokens and distinguish EOS from output-limit truncation. Define their loss treatment explicitly.
- Fail visibly on token/span inconsistencies or unexpectedly absent generated-token spans.
- Save sufficient token/configuration evidence for later loss auditing. Do not reconstruct historical raw token evidence and label it original.
- Test repeated text in system/user/tool messages, repeated assistant completions, tokenization boundaries, empty completions, EOS, and actual Qwen multi-turn template behavior.

## R2 — P1: The policy is not shown the complete public tool contract

Source: `src/cerl_rl/environment.py`, `tool_menu`.

The core supplies descriptions and schemas with argument types, required fields, enums, numeric constraints, and `additionalProperties`. The pilot drops these and displays only names and argument names, for example:

```text
- tickets__set_status(status, ticket_id)
```

It does not expose the accepted status values. The saved training evidence contains **four malformed attempts to set status to `closed`**, while the actual permitted values are `open`, `pending_customer`, `resolved`, and `escalated`.

This does not prove that exposing the schema would make the model succeed. It means the experiment should not interpret all such failures as inability to follow an adequately specified interface.

Required correction: render the existing public descriptions and schemas faithfully, including tool-specific constraints and requiredness. Keep privileged state and answers excluded. Recalculate the proposed prompt budget using that complete representation; the 3,639-token measurement from the abbreviated menu does not establish the size of the corrected prompt. Do not assume 8,192 tokens fits the Mac's memory merely because it fits the model's theoretical context window.

## R3 — P1: Sampling and differentiated probabilities differ

Source: `src/cerl_rl/rollout.py`, sampling arguments; `src/cerl_rl/grpo.py`, `group_backward`.

The code explicitly applies `temperature=1.0` and `top_p=0.95`; the pinned model configuration supplies `top_k=20`. The loss differentiates the ordinary unfiltered softmax. Both top-k and top-p alter the sampling distribution.

Setting only `top_k=0` leaves nucleus filtering active. A straightforward next-protocol option is temperature 1, top-k disabled, top-p 1, with all other probability-changing defaults explicitly neutralized and recorded. Another option is a mathematically specified objective that accounts for the actual behavior distribution. Do not casually substitute importance weights without considering the support removed by truncation.

One update per group can justify the local gradient of a ratio-free, advantage-weighted objective under appropriate matching conditions. It does not establish matching distributions or contexts; R1 and R3 violate those conditions. The claim that the bias is "small" is unmeasured and should be removed.

Document the actual single-update objective and its normalization rather than relying on the label GRPO. Also check the report's assertion that a per-episode token mean is a departure from the original GRPO formulation against the precise reference being used.

## R4 — P2: Arguments can override the selected tool

Source: `src/cerl_rl/environment.py`, `parse_action`.

The expression `{"kind": kind, **arguments}` permits a supplied argument named `kind` to replace the discriminator derived from `tool`. Independently reproduced input:

```json
{"tool":"tickets__get","arguments":{"kind":"billing.delete_customer","customer_id":"cus_000000000001"}}
```

Actual parsed result: `BillingDeleteCustomer`, rather than a malformed action. The core still enforces its ordinary backend and policy semantics; this finding is about the advertised action contract, not a claim of bypassing those interlocks.

Additionally, `payload.get("arguments") or {}` silently converts falsey non-object values to an empty object, and `tool_name_to_kind` itself is only a string replacement rather than an allowlist lookup.

Required correction: validate the selected tool against the public allowlist, reject reserved discriminator arguments and invalid argument-container types, and validate against that tool's actual schema/action type. Keep malformed handling distinct from blocked and committed violations. Check shared parsing code for the same pattern and report any necessary core change explicitly.

## R5 — P2: The new replay verifier accepts fabricated results

Source: `src/cerl_rl/replay.py`, `verify_run`.

The function checks only trace-head hashes. Although `replay_episode` computes additional fields, `verify_run` never compares them.

In a temporary copy, a baseline episode's reward was changed to 999, task completion to 1, safe completion to true, and terminal state hash to 64 `f` characters. The command returned:

```json
{"episodes_replayed":58,"mismatches":[],"ok":true}
```

An empty `{}` record returned `ok: true` with zero episodes. Missing `actions` silently skips a row.

The supplied original rewards are reproducible; the finding is that this command would not detect altered claims. Strengthen it to validate structure and required fields, replay and compare all claimed reproducible metrics/hashes, and check aggregates, group rewards, advantages, counts, and selections for consistency. Distinguish fields that cannot be reconstructed from actions, such as gradients and raw token sampling provenance. Partial/empty runs need an explicit status instead of a generic verified-success label.

## R6 — P2: Tool-call metrics count terminal declarations

Source: `src/cerl_rl/rollout.py`, counters and `Episode.tool_calls`.

Every non-malformed action increments `tool_calls`, including `finish`, `escalate`, and `abstain`. This disagrees with the benchmark verifier's tool-call definition.

| Phase | Reported tool calls | Replayed core tool calls |
|---|---:|---:|
| Baseline | 15 | **10** |
| After | 15 | **10** |
| Training | 191 | **150** |

There were 217 training actions in total: 150 actual tool calls, 41 terminal declarations, and 26 malformed actions. Report these categories separately. Derive benchmark metrics from the existing verdict rather than redefining them in the runner. Preserve historical rows and publish corrections in a derived audit/versioned result, rather than editing original evidence in place.

## R7 — P2: The deadline is advisory, and run persistence is incomplete

Source: `src/cerl_rl/pilot.py`, `Deadline`, `main`, `evaluate`, `_write`.

The deadline uses `time.time` and is checked only between groups. Baseline evaluation, each generation inside a group, backward passes, and after-evaluation are unbounded by it. Reserving eight minutes does not enforce the reserve.

A synthetic-clock test with zero requested updates and a 60-second budget advanced the clock by 100 seconds in each fake evaluation. After-evaluation started after expiration, and the program exited successfully recording 200 seconds. No real model was used and no actual delay was introduced.

The historical run did finish within the stated overall allowance. That measurement should remain. The implementation nevertheless cannot guarantee the advertised bound for a subsequent run.

`_write` writes directly over the record; checkpoints are saved only after the training loop; default output paths point at the original evidence directory. A failed or repeated run can lose the newest in-memory work or overwrite prior evidence.

Required correction: use a monotonic deadline across all phases with an enforceable bounded-worker mechanism or equivalent cancellation strategy; preserve completed work on interruption; write records/checkpoints atomically at sensible boundaries; refuse accidental reuse of an existing run directory. Test delayed/stalled fake workers and interruption paths without running the language model. Resume must retain the intended optimizer/RNG/budget state if exact resume is claimed.

## R8 — P2: CPU-only mathematical tests are not portable as packaged

Source: `src/cerl_rl/grpo.py`, unconditional `torch.mps.empty_cache()`.

The seven unmodified test failures are reproducible on Linux CPU PyTorch because the MPS backend is absent. Guard the cache operation by device/backend capability. The production pilot may remain MPS-only; the documented small CPU arithmetic checks should work without MPS.

The supplied arithmetic comparisons all passed after a diagnostic no-op replacement for that cache-maintenance call. This is an isolated portability defect, not evidence that the optimized masked-head arithmetic itself is wrong.

## R9 — Reporting and previously identified feasibility corrections

- The gold replays establish that the **reference trajectories** exceeded the old limits. They do not prove the scenarios were impossible for every policy, nor a hard maximum possible score of 2/5. Remove those claims in the report, protocol comments, and test documentation.
- The proposed 24-action limit accommodates the recorded gold workflows with headroom. Verify prompt/output feasibility again after R2 restores the public schemas. The `NEXT_*` constants are proposals; the executable still uses the historical constants.
- Context exhaustion and action exhaustion are currently combined under `step_limited`; preserve distinct termination reasons in future records.
- The protocol comment says an efficiency reward discourages waste, but the actual scalar does not weight `r_efficiency`. Correct the comment without silently changing the reward.
- The fact that no regularizer was used does not prove gradients came only from correctly identified generated tokens: R1 must be addressed before that stronger claim is made.
- Keep the observed 8 reward-weighted updates and 0/5 before/after result. Mark algorithmic limitations explicitly. The local pilot belongs to Phase 2; the broader research agenda remains incomplete.

## Minimal reproductions

Run from a checkout containing this pilot and the canonical simulator/corpus, using its isolated torch environment. These checks do not run a pretrained language model. Tokenization requires only the pinned tokenizer files.

### Incorrect mask source and changed conditioning

```python
import torch
from transformers import AutoTokenizer
from cerl_rl.rollout import _generated_mask, _messages

tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
answer = '{"tool": "tickets__get", "arguments": {"ticket_id": "tkt_000000000001"}}'
messages = [
    {"role": "system", "content": "An earlier answer was:\n" + answer},
    {"role": "user", "content": "Observation"},
    {"role": "assistant", "content": answer},
]
full = tok.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=False, enable_thinking=False)
ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
mask = _generated_mask(tok, ids, [answer])
system_end = full.index("<|im_end|>")
system_tokens = len(tok(full[:system_end], add_special_tokens=False)["input_ids"])
print("tokens incorrectly selected in system:", int(mask[:system_tokens].sum()))
# Observed: 32.

prompt = tok.apply_chat_template(_messages("System", ["First observation"], []),
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
complete = tok.apply_chat_template(
    _messages("System", ["First observation", "Second observation"],
              [answer, '{"tool":"abstain","arguments":{"reason":"x"}}']),
    tokenize=False, add_generation_prompt=False, enable_thinking=False)
generation_ids = tok(prompt, add_special_tokens=False)["input_ids"]
backward_ids = tok(complete[:complete.index(answer)], add_special_tokens=False)["input_ids"]
print("same first-action prefix:", generation_ids == backward_ids)
# Observed: False.
```

### Wrong tool dispatched

```python
from cerl_rl.environment import parse_action
print(parse_action(
    '{"tool":"tickets__get","arguments":'
    '{"kind":"billing.delete_customer","customer_id":"cus_000000000001"}}'
))
# Observed: BillingDeleteCustomer.
```

### Replay accepts altered metrics

```python
import json, tempfile
from pathlib import Path
from cerl_rl.replay import verify_run

record = json.loads(Path("evidence/rl-pilot/pilot_run.json").read_text())
record["baseline"]["per_episode"][0].update(
    reward=999, safe_completion=True, task_completion=1.0,
    terminal_state_hash="f" * 64,
)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "tampered.json"
    path.write_text(json.dumps(record))
    print(verify_run(path))
    path.write_text("{}")
    print(verify_run(path))
# Observed: ok=True in both cases.
```

## Next authorized work to request from Claude Code

Implement one offline correction pass covering R1-R9. Preserve the historical commits, corpus, adapter, and original evidence. Keep the corrected protocol and output records versioned separately. Use deterministic fake generations and tiny CPU networks, plus tokenizer-only checks, to establish correctness. Do not start model inference, model training, an LR sweep, or paid work during this pass.

Return a compact correction table linking each finding to a regression, exact offline results, the new local commit, and a review archive with checksums. Once those gates hold, the next model-dependent task should be a separately bounded checkpoint/logit diagnostic, not an automatic longer training run.
