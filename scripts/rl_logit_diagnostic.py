"""Does the trained adapter change the model's next-token prediction?

A deliberately small question, asked directly. The pilot showed identical greedy
behaviour before and after training, and the report offered several untested
explanations. One of them -- that the parameter change was too small in *effect*
to reorder the top token -- is measurable with four forward passes and no
training, and that is all this does.

Four stages on one fixed input:

1. adapter disabled
2. adapter disabled again          -> is the forward pass repeatable at all?
3. adapter enabled (in-memory, as trained)
4. adapter reloaded from disk      -> does the saved checkpoint reproduce (3)?

Stage 2 is the control. Without it, any difference in stage 3 could be
bf16/MPS run-to-run noise rather than the adapter, and the whole comparison
would be unreadable.

Partial results are written after every stage, so an external kill still leaves
whatever completed.

Not training. No optimizer, no episode, no sampling, no download.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, "src")

from cerl.agents.prompt_only import SYSTEM_PROMPT
from cerl.env.env import CerlEnv
from cerl.env.render import render_observation
from cerl_rl import protocol, protocol_v2
from cerl_rl.rollout_v2 import system_prompt_v2

ADAPTER = pathlib.Path("evidence/rl-pilot/adapter")
OUT = pathlib.Path("evidence/rl-logit-diagnostic/diagnostic.json")
TOP_K = 5

record: dict = {"stages": {}, "status": "started"}


def save() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, default=str) + "\n")


def checksum(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    started = time.monotonic()

    # -- 0. verify the checkpoint before touching a model ------------------
    weights = ADAPTER / "adapter_model.safetensors"
    recorded = json.loads(
        pathlib.Path("evidence/rl-pilot/pilot_run.json").read_text(),
    )["checkpoint"]["sha256"]
    digest = checksum(weights)
    record["adapter"] = {"path": str(weights), "sha256": digest,
                         "recorded_sha256": recorded, "match": digest == recorded}
    if digest != recorded:
        record["status"] = "aborted: adapter checksum mismatch"
        save()
        return 1
    save()

    # -- 1. the fixed input, built before any model work -------------------
    scenario_id = protocol_v2.training_selection().scenario_ids[0]
    from cerl.scenario import freeze

    scenario = freeze.load(freeze.FROZEN_DIR / f"{scenario_id}.json")
    env = CerlEnv(scenario)
    observation = env.reset()
    system = system_prompt_v2(SYSTEM_PROMPT)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION, local_files_only=True,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": render_observation(observation)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    record["input"] = {
        "scenario_id": scenario_id,
        "branch_withheld": "the policy input carries no branch label",
        "prompt_tokens": int(input_ids.shape[1]),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "first_16_token_ids": [int(t) for t in input_ids[0][:16]],
        "last_16_token_ids": [int(t) for t in input_ids[0][-16:]],
        "note": "identical tokens are reused for every stage; nothing is re-rendered",
    }
    save()

    # -- 2. the model ------------------------------------------------------
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION,
        dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER)).to(device)
    model.eval()
    ids = input_ids.to(device)
    record["environment"] = {
        "device": str(device), "dtype": "bfloat16",
        "model": protocol.MODEL_ID, "revision": protocol.MODEL_REVISION,
        "load_seconds": round(time.monotonic() - started, 1),
    }
    save()

    @torch.no_grad()
    def next_token_logits() -> torch.Tensor:
        """Logits for the position that predicts the next token. float32 for
        comparison only; the forward runs in the same bf16 the pilot used."""
        return model(input_ids=ids).logits[0, -1].float().cpu()

    def stage(name: str, logits: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, TOP_K)
        record["stages"][name] = {
            "argmax_token_id": int(logits.argmax()),
            "argmax_token": tokenizer.decode([int(logits.argmax())]),
            "max_logit": float(logits.max()),
            "logit_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            "top_5": [
                {"token_id": int(i), "token": tokenizer.decode([int(i)]),
                 "probability": float(p)}
                for p, i in zip(top.values, top.indices, strict=True)
            ],
        }
        save()

    # 1 & 2: adapter disabled, twice -- the repeatability control
    with model.disable_adapter():
        base_a = next_token_logits()
        stage("1_base_first", base_a)
        base_b = next_token_logits()
        stage("2_base_repeat", base_b)

    # 3: adapter enabled, as trained
    tuned = next_token_logits()
    stage("3_adapter_enabled", tuned)

    # 4: adapter reloaded from disk
    del model
    base2 = AutoModelForCausalLM.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION,
        dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    )
    model = PeftModel.from_pretrained(base2, str(ADAPTER)).to(device)
    model.eval()
    reloaded = next_token_logits()
    stage("4_adapter_reloaded", reloaded)

    # -- 3. comparisons ----------------------------------------------------
    def compare(label: str, a: torch.Tensor, b: torch.Tensor) -> dict:
        delta = (b - a)
        pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
        return {
            "identical_bits": bool(torch.equal(a, b)),
            "max_abs_logit_delta": float(delta.abs().max()),
            "mean_abs_logit_delta": float(delta.abs().mean()),
            "argmax_a": int(a.argmax()), "argmax_b": int(b.argmax()),
            "preferred_token_changed": int(a.argmax()) != int(b.argmax()),
            "top1_probability_a": float(pa.max()), "top1_probability_b": float(pb.max()),
            "kl_a_to_b": float((pa * (pa.clamp_min(1e-12).log()
                                      - pb.clamp_min(1e-12).log())).sum()),
            "label": label,
        }

    record["comparisons"] = {
        "base_repeatability": compare("stage 1 vs 2 -- same weights, same input", base_a, base_b),
        "adapter_effect": compare("stage 1 vs 3 -- adapter disabled vs enabled", base_a, tuned),
        "checkpoint_reproducibility": compare(
            "stage 3 vs 4 -- in-memory vs reloaded from disk", tuned, reloaded,
        ),
    }
    record["total_seconds"] = round(time.monotonic() - started, 1)
    record["status"] = "complete"
    save()
    print(json.dumps(record["comparisons"], indent=2))
    return 0


def _guarded() -> int:
    """Record why we stopped, whatever stopped us, before propagating.

    A SIGKILL from the watchdog cannot be caught -- that is why every stage
    saves as it completes -- but anything catchable is labelled here so a
    partial file says what happened rather than merely ending early.
    """
    try:
        return main()
    except BaseException as error:
        record["status"] = f"interrupted: {type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(_guarded())
