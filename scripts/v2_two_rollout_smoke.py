"""Two-rollout ENGINEERING SMOKE TEST of the v2 backward and optimizer path.

**What this is not**, stated first because it would be easy to misread:

* It is **not** a recovery of the original four-rollout update. That group is
  rollouts 0-3; this uses two of them, so it is a different group with different
  advantages and a different gradient.
* It is **not** a research result. The two rollouts were chosen **after**
  inspecting the previous run -- they are the two cheapest of the four, picked
  because the full group's 36 backward passes over 80,279 tokens did not fit in
  20 minutes. Selecting the cheap half after seeing the cost is a legitimate
  engineering decision and an illegitimate experimental one.
* It measures **nothing about task performance.** Task performance is left
  explicitly unmeasured: no evaluation is run before or after.

What it does test: whether ``turn_backward`` plus one optimizer step completes at
all on real weights, with real recorded token spans, inside the budget.

Inputs are the verified token records and replay-derived rewards of rollouts 0
and 1 from the trial at ``6f361d90``; the four-rollout evidence is untouched.
No new generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, "src")

TRIAL = pathlib.Path("evidence/rl-v2-trial2")
OUT = pathlib.Path("evidence/rl-v2-smoke-2rollout")
RECORD = OUT / "smoke.json"
PROGRESS = OUT / "backward_progress.jsonl"

#: The seed the recovery recorded, reused so the adapter is identical to it.
SEED = 20260908
#: Chosen after inspecting the previous run -- see the module docstring.
SELECTED = (0, 1)

record: dict = {
    "what_this_is": "engineering smoke test of the v2 backward + optimizer path",
    "what_this_is_not": [
        "not a recovery of the original four-rollout update (that group is 0-3)",
        "not a research performance result",
        "task performance is left explicitly unmeasured",
    ],
    "selection": {
        "rollouts": list(SELECTED),
        "rule": "the first two by original order",
        "when_chosen": "AFTER inspecting the previous run",
        "why": ("the full four-rollout group needed 36 backward passes over 80,279 "
                "tokens and did not fit in 20 minutes; these two are the cheapest "
                "of the four. Choosing the cheap half after seeing the cost is an "
                "engineering decision, not an experimental one"),
    },
    "source_trial": "6f361d90b593c1e749a43dbd8c026135bed818c0",
    "status": "started",
}


def save() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = RECORD.with_suffix(".json.tmp")
    with scratch.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    scratch.replace(RECORD)


def note_progress(entry: dict) -> None:
    """Append and fsync after every completed turn backward."""
    entry["t"] = round(time.monotonic() - START, 2)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


START = time.monotonic()


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "recover", "scripts/recover_v2_update.py",
    )
    recover = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recover)

    ok, data = recover.check_preconditions()
    record["preconditions_on_all_four"] = data["summary"]
    save()
    if not ok:
        record["status"] = "STOPPED: preconditions failed on the recorded evidence"
        save()
        return 2

    import torch

    from cerl_rl import protocol_v2
    from cerl_rl.grpo import advantages_for, turn_backward
    from cerl_rl.model import adapter_state, load_policy, pick_device

    selected = {i: data["episodes"][i] for i in SELECTED}
    record["group"] = {
        "rollouts": list(SELECTED),
        "turns": sum(len(e["turns"]) for e in selected.values()),
        "generated_tokens": sum(
            sum(len(t["generated_ids"]) for t in e["turns"]) for e in selected.values()
        ),
        "total_sequence_tokens": sum(
            len(t["prompt_ids"]) + len(t["generated_ids"])
            for e in selected.values() for t in e["turns"]
        ),
        "rewards": [selected[i]["reward"] for i in SELECTED],
    }
    save()

    rewards = [selected[i]["reward"] for i in SELECTED]
    advantages = advantages_for(rewards)
    record["group"]["advantages"] = advantages
    record["group"]["note"] = (
        "advantages are recomputed for THIS two-rollout group; they are not the "
        "four-rollout group's advantages"
    )
    save()
    if advantages is None:
        record["status"] = "SKIPPED: no within-group reward variation"
        save()
        return 0

    torch.manual_seed(SEED)
    device = pick_device()
    model = load_policy(device, lora_rank=8)
    model.eval()
    record["config"] = {
        "seed": SEED, "device": str(device), "lora_rank": 8,
        "learning_rate": 1e-5, "objective": protocol_v2.REWARD_NAME,
        "kl_beta": 0.0, "weight_decay": 0.0,
        "loss": "cerl_rl.grpo.turn_backward",
        "load_seconds": round(time.monotonic() - START, 1),
    }
    b_zero = all(
        float(p.detach().abs().sum()) == 0.0
        for n, p in model.named_parameters() if "lora_B" in n
    )
    record["config"]["all_lora_B_zero_before"] = b_zero
    save()

    episodes = recover.build_episodes(selected)
    before = adapter_state(model)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)

    PROGRESS.write_text("")
    backward_started = time.monotonic()
    loss, tokens = turn_backward(model, episodes, advantages, device,
                                 on_turn_done=note_progress)
    record["backward"] = {
        "completed": True, "loss": loss, "scored_tokens": tokens,
        "seconds": round(time.monotonic() - backward_started, 1),
        "turns_backwarded": sum(1 for _ in PROGRESS.read_text().splitlines()),
    }
    save()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0
    finite = bool(torch.isfinite(torch.tensor(grad_norm)))
    record["gradient"] = {"norm": grad_norm, "finite": finite,
                          "tensors_with_grad": len(grads)}
    save()
    if not finite or grad_norm == 0.0:
        record["update"] = {"occurred": False}
        record["status"] = (
            f"SKIPPED: gradient is {'non-finite' if not finite else 'exactly zero'}"
        )
        save()
        return 0

    optimizer.step()
    after = adapter_state(model)
    delta = {k: (after[k] - before[k]).abs() for k in before}
    record["update"] = {
        "occurred": True, "optimizer": "AdamW", "learning_rate": 1e-5,
        "l1": sum(float(d.sum()) for d in delta.values()),
        "linf": max(float(d.max()) for d in delta.values()),
        "tensors_changed": sum(1 for d in delta.values() if float(d.max()) > 0),
        "tensors_total": len(delta),
        "lora_B_changed": sum(1 for k, d in delta.items()
                              if "lora_B" in k and float(d.max()) > 0),
        "lora_A_changed": sum(1 for k, d in delta.items()
                              if "lora_A" in k and float(d.max()) > 0),
    }
    save()

    checkpoint = OUT / "adapter"
    model.save_pretrained(checkpoint)
    weights = next(checkpoint.glob("adapter_model*"), None)
    record["checkpoint"] = {
        "path": str(checkpoint), "weights_file": weights.name if weights else None,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None,
        "bytes": weights.stat().st_size if weights else None,
        "atomic": "written by save_pretrained, then checksummed from disk",
    }
    record["task_performance"] = (
        "EXPLICITLY UNMEASURED. No evaluation was run before or after this update. "
        "A completed update shows the machinery runs; it says nothing about whether "
        "the policy improved."
    )
    record["total_seconds"] = round(time.monotonic() - START, 1)
    record["status"] = "completed: one optimizer step on a two-rollout smoke group"
    save()
    print(json.dumps({k: record[k] for k in
                      ("status", "group", "backward", "gradient", "update", "checkpoint")},
                     indent=2))
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
