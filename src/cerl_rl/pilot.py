"""The bounded pilot: baseline, GRPO training, and the same evaluation again.

Every number this writes is measured. Nothing is defaulted, inferred from a
config, or carried over from a previous run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time
from typing import Any

import torch

from cerl.agents.prompt_only import SYSTEM_PROMPT
from cerl_rl import protocol
from cerl_rl import rollout as rollout_module
from cerl_rl.grpo import GroupStats, advantages_for, group_backward
from cerl_rl.model import adapter_state, load_policy, load_tokenizer, pick_device

OUT = pathlib.Path("evidence/rl-pilot")


class Deadline:
    """A wall-clock stop, checked between units of work.

    Enforced outside the training loop so a run cannot talk itself into "just
    one more update". Partial results are always written: an interrupted pilot
    that reported nothing would be worse than one that reported less.
    """

    def __init__(self, seconds: float) -> None:
        self.started = time.time()
        self.limit = seconds

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def remaining(self) -> float:
        return self.limit - self.elapsed

    def expired(self, reserve: float = 0.0) -> bool:
        return self.remaining <= reserve


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    for part in out.split():
        if part.startswith("used"):
            continue
    try:
        return float(out.split("used = ")[1].split("M")[0])
    except Exception:  # noqa: BLE001
        return -1.0


def system_prompt() -> str:
    return SYSTEM_PROMPT + rollout_module.SYSTEM_SUFFIX % rollout_module.tool_menu()


def evaluate(
    model: Any, tokenizer: Any, scenarios: list[Any], device: Any, *, label: str,
) -> dict[str, Any]:
    """The before/after measurement. One function, one configuration, both times.

    Greedy and seeded, so a difference between the two calls is a difference in
    the policy and not in the sampling.
    """
    model.eval()
    torch.manual_seed(protocol.EVAL_SEED)
    episodes = []
    started = time.time()
    for scenario in scenarios:
        tokenised = rollout_module.rollout(
            model, tokenizer, scenario,
            max_actions=protocol.MAX_ACTIONS,
            max_new_tokens=protocol.MAX_NEW_TOKENS,
            temperature=protocol.EVAL_TEMPERATURE,
            device=device,
            system_prompt=system_prompt(),
            max_prompt_tokens=protocol.MAX_PROMPT_TOKENS,
        )
        episodes.append(tokenised.episode)
        torch.mps.empty_cache()
    n = len(episodes)
    return {
        "label": label,
        "seconds": round(time.time() - started, 1),
        "episodes": n,
        "task_completion_mean": sum(e.task_completion for e in episodes) / n,
        "safe_completion": sum(1 for e in episodes if e.safe_completion),
        "decision_correct": sum(1 for e in episodes if e.decision_correct),
        "reward_mean": sum(e.reward for e in episodes) / n,
        "committed_violations": sum(len(e.committed_violations) for e in episodes),
        "attempted_violations": sum(len(e.attempted_violations) for e in episodes),
        "tool_calls": sum(e.tool_calls for e in episodes),
        "malformed_actions": sum(e.malformed_actions for e in episodes),
        "step_limited": sum(1 for e in episodes if e.step_limited),
        "per_episode": [
            {
                "scenario_id": e.scenario_id,
                "reward": e.reward,
                "task_completion": e.task_completion,
                "safe_completion": e.safe_completion,
                "decision_correct": e.decision_correct,
                "declared_outcome": e.declared_outcome,
                "failure_class": e.failure_class,
                "committed": list(e.committed_violations),
                "attempted": list(e.attempted_violations),
                "tool_calls": e.tool_calls,
                "malformed_actions": e.malformed_actions,
                "step_limited": e.step_limited,
                "turns": len(e.turns),
                "terminal_state_hash": e.terminal_state_hash,
                "trace_head_hash": e.trace_head_hash,
                "actions": [a.model_dump(mode="json") for a in e.actions],
            }
            for e in episodes
        ],
    }


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: path.open("rb").read(1 << 20), b""):  # pragma: no cover
        h.update(chunk)
        break
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--deadline-minutes", type=float, default=105.0)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    deadline = Deadline(args.deadline_minutes * 60)
    record: dict[str, Any] = {
        "protocol": protocol.manifest(),
        "config": {
            "updates_requested": args.updates,
            "group_size": args.group_size,
            "learning_rate": args.lr,
            "lora_rank": args.lora_rank,
            "kl_beta": 0.0,
            "kl_note": (
                "No KL/entropy term and no weight decay: with zero regularization "
                "every parameter change is attributable to the reward objective "
                "alone, which is what this pilot has to be able to claim."
            ),
            "deadline_minutes": args.deadline_minutes,
        },
        "device": "mps",
        "groups": [],
        "stopped_because": None,
    }

    device = pick_device()
    tokenizer = load_tokenizer()
    model = load_policy(device, lora_rank=args.lora_rank)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record["trainable_parameters"] = trainable

    train_scenarios = protocol.training_selection().load()
    val_scenarios = protocol.validation_selection().load()

    # -- baseline ----------------------------------------------------------
    record["baseline"] = evaluate(model, tokenizer, val_scenarios, device, label="baseline")
    _write(args.out, record)

    initial = adapter_state(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0,
    )

    reward_driven = 0
    for update in range(args.updates):
        if deadline.expired(reserve=8 * 60):   # keep time for the after-evaluation
            record["stopped_because"] = f"deadline reached after {update} updates"
            break
        scenario = train_scenarios[update % len(train_scenarios)]
        model.eval()
        torch.manual_seed(protocol.EVAL_SEED + update)
        episodes = []
        t0 = time.time()
        for _ in range(args.group_size):
            episodes.append(
                rollout_module.rollout(
                    model, tokenizer, scenario,
                    max_actions=protocol.MAX_ACTIONS,
                    max_new_tokens=protocol.MAX_NEW_TOKENS,
                    temperature=protocol.TRAIN_TEMPERATURE,
                    device=device,
                    system_prompt=system_prompt(),
                    max_prompt_tokens=protocol.MAX_PROMPT_TOKENS,
                ),
            )
            torch.mps.empty_cache()
        gen_seconds = time.time() - t0

        rewards = [t.episode.reward for t in episodes]
        stats = GroupStats(scenario.scenario_id, rewards)
        advantages = advantages_for(rewards)
        if advantages is None:
            # Recorded, not silently counted as an update. A zero-advantage group
            # would move parameters only through the optimizer's own state.
            stats.skipped = True
            stats.skip_reason = "no within-group reward variation; advantages would be zero"
            entry = stats.as_dict()
            entry.update(update=update, generation_seconds=round(gen_seconds, 1))
            entry["episodes"] = [_episode_row(t) for t in episodes]
            record["groups"].append(entry)
            _write(args.out, record)
            continue

        stats.advantages = advantages
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t1 = time.time()
        loss, tokens = group_backward(model, episodes, advantages, device)
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))
        if not torch.isfinite(torch.tensor(grad_norm)):
            record["stopped_because"] = f"non-finite gradient at update {update}"
            break
        optimizer.step()
        stats.loss, stats.grad_norm, stats.tokens = loss, grad_norm, tokens
        reward_driven += 1

        entry = stats.as_dict()
        entry.update(
            update=update,
            generation_seconds=round(gen_seconds, 1),
            backward_seconds=round(time.time() - t1, 1),
            swap_used_mb=swap_used_mb(),
            mps_gb=round(torch.mps.driver_allocated_memory() / 1e9, 2),
        )
        entry["episodes"] = [_episode_row(t) for t in episodes]
        record["groups"].append(entry)
        record["reward_driven_updates"] = reward_driven
        _write(args.out, record)

    record.setdefault("reward_driven_updates", reward_driven)
    record["stopped_because"] = record["stopped_because"] or "completed all requested updates"

    # -- what actually changed --------------------------------------------
    final = adapter_state(model)
    delta = {k: (final[k] - initial[k]).abs() for k in initial}
    record["adapter_change"] = {
        "l1": sum(float(d.sum()) for d in delta.values()),
        "linf": max(float(d.max()) for d in delta.values()),
        "tensors_changed": sum(1 for d in delta.values() if float(d.max()) > 0),
        "tensors_total": len(delta),
    }

    ckpt = args.out / "adapter"
    model.save_pretrained(ckpt)
    weights = next(ckpt.glob("adapter_model*"), None)
    record["checkpoint"] = {
        "path": str(ckpt),
        "weights_file": weights.name if weights else None,
        "sha256": sha256_of(weights) if weights else None,
    }

    record["after"] = evaluate(model, tokenizer, val_scenarios, device, label="after")
    record["total_model_seconds"] = round(deadline.elapsed, 1)
    _write(args.out, record)
    summary = {k: record[k] for k in ("reward_driven_updates", "stopped_because")}
    print(json.dumps(summary, indent=2))
    return 0


def _episode_row(t: Any) -> dict[str, Any]:
    e = t.episode
    return {
        "reward": e.reward,
        "task_completion": e.task_completion,
        "safe_completion": e.safe_completion,
        "declared_outcome": e.declared_outcome,
        "failure_class": e.failure_class,
        "committed": list(e.committed_violations),
        "attempted": list(e.attempted_violations),
        "malformed_actions": e.malformed_actions,
        "tool_calls": e.tool_calls,
        "step_limited": e.step_limited,
        "generated_tokens": t.n_generated_tokens,
        "total_tokens": int(t.input_ids.numel()),
        "actions": [a.model_dump(mode="json") for a in e.actions],
        "trace_head_hash": e.trace_head_hash,
    }


def _write(out: pathlib.Path, record: dict[str, Any]) -> None:
    (out / "pilot_run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
