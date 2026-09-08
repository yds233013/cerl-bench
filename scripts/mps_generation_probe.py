"""Isolate the Metal abort that killed the v2 trial. One probe per process.

The trial died with::

    MPSNDArray.mm:788 failed assertion `[MPSTemporaryNDArray
    initWithDevice:descriptor:] Error: total bytes of NDArray > 2**32'

inside the first ``generate``. That is an assertion, not an exception: it aborts
the process, so nothing in-process can catch it and report. The only way to
learn where it happens is to leave a durable trail and see which marker is last.

So each probe writes JSONL markers that are flushed and ``fsync``ed before and
after every model forward and every sampling operation, and calls
``torch.mps.synchronize()`` at those boundaries -- MPS work is queued
asynchronously, so without a synchronize the last marker would say nothing about
where the failure actually was.

Each probe runs in its own process (see ``run_mps_probes.sh``) so an abort takes
down only that probe. No training, no episode, no download.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

import torch

sys.path.insert(0, "src")

OUT = pathlib.Path("evidence/mps-probe")
VOCAB = 151936          # Qwen3-0.6B, checked against config in probe 1
LIMIT = 2**32           # the MPS single-allocation ceiling in the assertion


class Trail:
    """Append-only, fsynced stage markers. The last one written is the truth."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.started = time.monotonic()
        path.write_text("")

    def mark(self, stage: str, **fields: object) -> None:
        payload = {"t": round(time.monotonic() - self.started, 3), "stage": stage, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def sync(self, stage: str, **fields: object) -> None:
        """Mark, then wait for the GPU so the marker means something."""
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.mark(stage, synchronized=True, **fields)


def describe(tensor: torch.Tensor) -> dict[str, object]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "bytes": tensor.element_size() * tensor.nelement()}


# --------------------------------------------------------------------------
# probe 1 -- the sampling operation alone, no pretrained weights
# --------------------------------------------------------------------------


def probe_1(trail: Trail) -> None:
    """What does the installed generation code do, and does its sampling op
    survive a real-vocabulary tensor on MPS? No weights are loaded."""
    import transformers
    from transformers import GenerationConfig

    from cerl_rl import protocol, protocol_v2

    trail.mark("versions", transformers=transformers.__version__,
               torch=torch.__version__, mps=torch.backends.mps.is_available())

    config = GenerationConfig.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION, local_files_only=True,
    )
    resolved = config.to_dict()
    merged = {**resolved, "do_sample": True, "temperature": protocol_v2.TRAIN_TEMPERATURE,
              **dict(protocol_v2.SAMPLING_NEUTRALISED)}
    trail.mark("resolved_generation_config",
               file={k: resolved.get(k) for k in
                     ("do_sample", "temperature", "top_k", "top_p", "min_p",
                      "repetition_penalty", "eos_token_id", "pad_token_id")},
               merged_for_v2={k: merged.get(k) for k in
                              ("do_sample", "temperature", "top_k", "top_p", "min_p",
                               "typical_p", "repetition_penalty", "no_repeat_ngram_size",
                               "renormalize_logits")})

    # Which warpers would this configuration actually construct?
    trail.mark("logits_processor_api",
               present=hasattr(transformers.GenerationMixin, "_get_logits_processor"))
    active = [name for name, value in
              (("temperature", merged["temperature"]), ("top_k", merged["top_k"]),
               ("top_p", merged["top_p"]), ("min_p", merged["min_p"]))
              if not (
                  (name == "temperature" and value == 1.0)
                  or (name == "top_k" and value in (0, None))
                  or (name == "top_p" and value == 1.0)
                  or (name == "min_p" and value in (0.0, None))
              )]
    trail.mark("active_warpers_under_v2_settings", warpers=active or ["none"])

    device = torch.device("mps")
    for dtype in (torch.float32,):
        logits = torch.randn(1, VOCAB, dtype=dtype, device=device)
        trail.sync("synthetic_logits_built", **describe(logits))

        trail.mark("softmax_begin")
        probs = torch.softmax(logits, dim=-1)
        trail.sync("softmax_done", **describe(probs))

        trail.mark("multinomial_begin", note="this is what generate() calls when sampling")
        picked = torch.multinomial(probs, num_samples=1)
        trail.sync("multinomial_done", **describe(picked), value=int(picked.item()))

        trail.mark("argmax_begin")
        best = torch.argmax(logits, dim=-1)
        trail.sync("argmax_done", **describe(best), value=int(best.item()))

        trail.mark("sort_begin", note="top-p would sort the vocabulary")
        ordered, _index = torch.sort(logits, descending=True)
        trail.sync("sort_done", **describe(ordered))
    trail.mark("probe_1_complete", verdict="sampling ops survive a real-vocabulary tensor")


# --------------------------------------------------------------------------
# shared setup for the weight-loading probes
# --------------------------------------------------------------------------


def _policy_and_prompt(trail: Trail):
    from cerl.agents.prompt_only import SYSTEM_PROMPT
    from cerl.env.env import CerlEnv
    from cerl.env.render import render_observation
    from cerl.scenario import freeze
    from cerl_rl import protocol_v2
    from cerl_rl.model import load_policy, load_tokenizer, pick_device
    from cerl_rl.rollout_v2 import _messages, system_prompt_v2

    device = pick_device()
    trail.mark("loading_tokenizer")
    tokenizer = load_tokenizer()
    trail.mark("loading_policy", note="fresh rank-8 LoRA, bfloat16, as the trial used")
    model = load_policy(device, lora_rank=8)
    model.eval()
    trail.sync("policy_loaded")

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
    mask = encoded["attention_mask"].to(device)
    trail.mark("prompt_built", scenario_id=scenario_id, **describe(ids))
    return model, tokenizer, ids, mask, device


def _generate(trail: Trail, model, tokenizer, ids, mask, *, max_new_tokens: int,
              sampling: dict, label: str) -> None:
    from cerl_rl import protocol_v2

    trail.mark(f"{label}_generate_begin", max_new_tokens=max_new_tokens,
               sampling=sampling, prompt_tokens=int(ids.shape[1]))
    out = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **sampling,
    )
    trail.sync(f"{label}_generate_done", **describe(out),
               new_tokens=int(out.shape[1] - ids.shape[1]))
    _ = protocol_v2  # keep the import meaningful for readers


def probe_2(trail: Trail) -> None:
    """One token, exact v2 sampling settings."""
    from cerl_rl import protocol_v2

    model, tokenizer, ids, mask, _device = _policy_and_prompt(trail)
    _generate(trail, model, tokenizer, ids, mask, max_new_tokens=1,
              sampling={"do_sample": True, "temperature": protocol_v2.TRAIN_TEMPERATURE,
                        **dict(protocol_v2.SAMPLING_NEUTRALISED)}, label="v2_sampling")
    trail.mark("probe_2_complete")


def probe_3(trail: Trail) -> None:
    """One token, greedy. Only the sampling mode differs from probe 2."""
    model, tokenizer, ids, mask, _device = _policy_and_prompt(trail)
    _generate(trail, model, tokenizer, ids, mask, max_new_tokens=1,
              sampling={"do_sample": False}, label="greedy")
    trail.mark("probe_3_complete")


def probe_4(trail: Trail) -> None:
    """Targeted comparison, chosen by probes 1-3.

    Probe 1 showed the sampling primitives survive a real-vocabulary tensor.
    Probe 3 showed greedy generation on this exact prompt is fine. Probe 2
    aborted with ``do_sample=True`` plus the v2 settings at a single token. So
    the trigger is inside sampled generation, and the question is which
    argument introduces it.

    One process, several one-token generations, adding settings one at a time
    from plainest to most specific. Every case is marked before and after and
    the marker is fsynced, so when the process aborts the last recorded case is
    the one that did it -- an abort leaves no exception to catch.
    """
    from cerl_rl import protocol_v2

    model, tokenizer, ids, mask, _device = _policy_and_prompt(trail)

    neutral = dict(protocol_v2.SAMPLING_NEUTRALISED)
    cases: list[tuple[str, dict]] = [
        ("a_sample_defaults", {"do_sample": True}),
        ("b_sample_temperature", {"do_sample": True, "temperature": 1.0}),
        ("c_plus_top_k_0", {"do_sample": True, "temperature": 1.0, "top_k": 0}),
        ("d_plus_top_p_1", {"do_sample": True, "temperature": 1.0, "top_k": 0,
                            "top_p": 1.0}),
        ("e_plus_min_p_0", {"do_sample": True, "temperature": 1.0, "top_k": 0,
                            "top_p": 1.0, "min_p": 0.0}),
        ("f_full_v2", {"do_sample": True, "temperature": protocol_v2.TRAIN_TEMPERATURE,
                       **neutral}),
    ]
    for name, sampling in cases:
        trail.mark("case_begin", case=name, sampling=sampling)
        out = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **sampling,
        )
        trail.sync("case_done", case=name, new_tokens=int(out.shape[1] - ids.shape[1]))
    trail.mark("probe_4_complete", verdict="every sampling configuration survived")


PROBES = {1: probe_1, 2: probe_2, 3: probe_3, 4: probe_4}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=int, required=True, choices=sorted(PROBES))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    trail = Trail(OUT / f"probe_{args.probe}.jsonl")
    trail.mark("probe_begin", probe=args.probe, pid=os.getpid())
    try:
        PROBES[args.probe](trail)
    except BaseException as error:
        trail.mark("python_exception", type=type(error).__name__, detail=str(error)[:400],
                   traceback=traceback.format_exc()[-1200:])
        return 1
    trail.mark("probe_end", probe=args.probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
