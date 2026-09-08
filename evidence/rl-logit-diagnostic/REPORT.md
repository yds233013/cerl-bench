# Does the trained adapter change what the model predicts?

**Yes — measurably, but not enough to change its choice at the position tested.**

One fixed input, four forward passes, 35 seconds. No training, no optimizer, no
episode, no sampling, no download.

## What was run

`scripts/rl_logit_diagnostic.py`, under the external watchdog
`scripts/run_logit_diagnostic.sh` (10-minute hard limit covering all model work
including loading; it fired at 35.2 s, so nothing was cut short).

The adapter's checksum was verified **before** any model was loaded:
`96391951…41ff4`, matching the recorded pilot run.

The input is the first observation of the first canonical *training* scenario,
rendered with the complete v2 prompt — 1,524 tokens. The same token ids are
reused for every stage; nothing is re-rendered between them. Model
`Qwen/Qwen3-0.6B` at revision `c1899de…47ca`, bfloat16 on MPS, as the pilot ran.

| stage | what |
|---|---|
| 1 | adapter **disabled** |
| 2 | adapter **disabled again** — the repeatability control |
| 3 | adapter **enabled**, as trained |
| 4 | adapter **reloaded from disk** |

Stage 2 is what makes the rest readable. Without it, any difference in stage 3
could be bf16/MPS run-to-run noise rather than the adapter.

## Results

| comparison | identical? | max \|Δlogit\| | mean \|Δlogit\| | KL | preferred token |
|---|---|---|---|---|---|
| 1 vs 2 — repeatability | **bit-identical** | 0.0 | 0.0 | 0.0 | unchanged |
| 1 vs 3 — **adapter effect** | **different** | **0.8125** | 0.1138 | 2.17e−4 | **unchanged** |
| 3 vs 4 — reload | **bit-identical** | 0.0 | 0.0 | 0.0 | unchanged |

Top five next-token probabilities:

| token | base (stage 1) | with adapter (stage 3) |
|---|---|---|
| `{"` | **0.991146** | **0.989242** |
| `"` | 0.005894 | 0.007553 |
| ` ``` ` | 0.001913 | 0.002164 |
| `{\n` | 0.000798 | 0.000796 |
| `{` | 0.000108 | 0.000108 |

## What this establishes

1. **The forward pass is deterministic here.** Two runs with the same weights and
   the same input produced bit-identical logits, so the control holds and any
   difference below is attributable to the adapter.
2. **The adapter does change the output distribution.** Not identical: the
   largest logit moves by 0.8125. At a magnitude near 31, one bfloat16 step is
   0.125, so this is roughly **6.5 representable steps** — a real change, not a
   rounding artifact.
3. **It does not change the model's choice at this position.** The preferred
   token is `{"` before and after, at 0.991 → 0.989. The top token is not in a
   close race with the second (0.99 vs 0.006), so a shift of this size cannot
   reorder it.
4. **The saved checkpoint reproduces exactly.** The reloaded adapter gives
   bit-identical logits to the in-memory one, which closes the "disk reload
   equivalence" item previously marked UNVERIFIED.

## What this does **not** establish

- **It is not evidence of improved task performance.** A changed prediction is a
  changed prediction. Whether the change is in a useful direction is a different
  question that this cannot answer, and the pilot's 0/5 before and after stands.
- **One position, one input.** This is the first token of the first turn of one
  scenario. An episode has many decision points, some of them close races where a
  0.8 logit shift could flip the outcome. Nothing here rules that in or out.
- **It does not fully settle the earlier hypothesis.** The report offered "the
  update was too small in effect to reorder the top token" as one untested
  explanation for the unchanged validation behaviour. This measurement is
  *consistent* with it at this position and makes it more plausible, but a single
  position is not the episode, and the competing explanation — that the gradient
  direction was uninformative, given the reward was mostly the decision term on
  partly-unreachable episodes — is untouched by this.
- The adapter was trained under the v1 path, whose token provenance was later
  found unsound. This diagnostic measures what that adapter does; it does not
  validate how it was produced.

## Files

- `scripts/rl_logit_diagnostic.py` — the diagnostic; saves after every stage
- `scripts/run_logit_diagnostic.sh` — external watchdog
- `evidence/rl-logit-diagnostic/diagnostic.json` — full record, including logit
  hashes and top-5 tables for all four stages
- `evidence/rl-logit-diagnostic/stdout.log` — run log
