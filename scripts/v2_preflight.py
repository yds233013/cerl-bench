"""Preflight for the corrected v2 trial.

The previous trial died inside its first ``generate`` because ``min_p=0.0``
built the very warper it was meant to disable. The fix is to omit the key, and
this checks that the fix actually holds *in the resolved configuration* before
any training time is spent -- not by re-reading the constant, but by resolving
the generation config the way ``generate`` does, listing the logits processors
that would run, and then generating a single token on the exact prompt the trial
will use.

If any of that fails the trial does not start. Written as its own process so an
abort here leaves the trial's record untouched.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback

import torch

sys.path.insert(0, "src")

OUT = pathlib.Path("evidence/rl-v2-trial2")
RECORD = OUT / "preflight.json"

record: dict = {"status": "started", "checks": {}}


def save() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RECORD.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    started = time.monotonic()
    from transformers.generation.logits_process import LogitsProcessorList, MinPLogitsWarper

    from cerl.agents.prompt_only import SYSTEM_PROMPT
    from cerl.env.env import CerlEnv
    from cerl.env.render import render_observation
    from cerl.scenario import freeze
    from cerl_rl import protocol_v2
    from cerl_rl.model import load_policy, load_tokenizer, pick_device
    from cerl_rl.rollout_v2 import _messages, system_prompt_v2

    sampling = {"do_sample": True, "temperature": protocol_v2.TRAIN_TEMPERATURE,
                **dict(protocol_v2.SAMPLING_NEUTRALISED)}
    record["sampling_kwargs"] = sampling
    save()

    device = pick_device()
    tokenizer = load_tokenizer()
    model = load_policy(device, lora_rank=8)
    model.eval()
    record["load_seconds"] = round(time.monotonic() - started, 1)
    save()

    # -- 1. the fully resolved generation configuration --------------------
    resolved = model.generation_config.to_dict()
    resolved.update(sampling)
    record["resolved_generation_config"] = {
        k: resolved.get(k) for k in sorted(
            ("do_sample", "temperature", "top_k", "top_p", "min_p", "typical_p",
             "repetition_penalty", "no_repeat_ngram_size", "renormalize_logits",
             "max_new_tokens", "eos_token_id", "pad_token_id", "num_beams"),
        )
    }
    effective_min_p = resolved.get("min_p")
    record["checks"]["effective_min_p_is_none"] = effective_min_p is None
    save()
    if effective_min_p is not None:
        record["status"] = f"FAILED: effective min_p is {effective_min_p!r}, not None"
        save()
        return 1

    # -- 2. the logits processors that would actually run ------------------
    import inspect

    from transformers import GenerationConfig

    config = GenerationConfig.from_dict(resolved)
    signature = inspect.signature(model._get_logits_processor)
    kwargs: dict = {}
    for name in signature.parameters:
        if name == "generation_config":
            kwargs[name] = config
        elif name in {"input_ids_seq_length", "input_ids_length"}:
            kwargs[name] = 16
        elif name == "encoder_input_ids":
            kwargs[name] = torch.zeros(1, 1, dtype=torch.long, device=device)
        elif name == "prefix_allowed_tokens_fn":
            kwargs[name] = None
        elif name == "logits_processor":
            kwargs[name] = LogitsProcessorList()
        elif name == "device":
            kwargs[name] = device
        elif name == "model_kwargs":
            kwargs[name] = {}
    processors = model._get_logits_processor(**kwargs)
    names = [type(p).__name__ for p in processors]
    record["active_logits_processors"] = names
    record["checks"]["min_p_warper_absent"] = not any(
        isinstance(p, MinPLogitsWarper) for p in processors
    )
    save()
    if not record["checks"]["min_p_warper_absent"]:
        record["status"] = "FAILED: MinPLogitsWarper is still constructed"
        save()
        return 1

    # -- 3. one token on the exact initial prompt --------------------------
    scenario_id = protocol_v2.training_selection().scenario_ids[0]
    scenario = freeze.load(freeze.FROZEN_DIR / f"{scenario_id}.json")
    env = CerlEnv(scenario)
    observation = env.reset()
    prompt = tokenizer.apply_chat_template(
        _messages(system_prompt_v2(SYSTEM_PROMPT), [render_observation(observation)], []),
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(device)
    record["prompt"] = {"scenario_id": scenario_id, "tokens": int(ids.shape[1])}
    save()

    generation_started = time.monotonic()
    out = model.generate(
        input_ids=ids, attention_mask=encoded["attention_mask"].to(device),
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **sampling,
    )
    torch.mps.synchronize()
    record["checks"]["one_token_generation"] = {
        "passed": True,
        "seconds": round(time.monotonic() - generation_started, 1),
        "new_tokens": int(out.shape[1] - ids.shape[1]),
        "token": tokenizer.decode(out[0, ids.shape[1]:]),
    }
    record["seed_note"] = (
        "The trial runs in a separate process, so no RNG state from this "
        "preflight can reach it; the pilot seeds per group from "
        "protocol_v2.EVAL_SEED."
    )
    record["status"] = "passed"
    record["total_seconds"] = round(time.monotonic() - started, 1)
    save()
    print(json.dumps({"status": record["status"],
                      "checks": record["checks"],
                      "processors": names}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        record["status"] = f"FAILED: {type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()[-1500:]
        save()
        raise
