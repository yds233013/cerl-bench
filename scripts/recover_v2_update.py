"""Recover one training update from the rollouts the 6f361d trial already recorded.

That trial generated four complete rollouts and was then killed by its watchdog
before the optimizer ran. The rollouts survive with exact token provenance, so
the update can be attempted from the record instead of regenerating -- which
would be a different sample, not a recovery.

**No new generation.** Every token used here was produced by the original trial.

Preconditions are checked first and are refusals, not warnings: if any episode is
short of its original termination, or any turn is missing provenance, or replay
disagrees, the run stops and says what is missing. Training on a shortened group
would silently change the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, "src")

TRIAL = pathlib.Path("evidence/rl-v2-trial2")
OUT = pathlib.Path("evidence/rl-v2-recovery")
RECORD = OUT / "recovery.json"
TERMINAL = {"finish", "escalate", "abstain"}

record: dict = {
    "what_this_is": (
        "Recovery of one optimizer update from the four rollouts recorded by the "
        "trial at commit 6f361d90b593c1e749a43dbd8c026135bed818c0. No new "
        "generation: every token was produced by that trial."
    ),
    "source_trial": {"commit": "6f361d90b593c1e749a43dbd8c026135bed818c0",
                     "evidence": str(TRIAL)},
    "status": "started",
    "preconditions": {},
}


def save() -> None:
    """Atomic: a partial write can never replace a good record."""
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = RECORD.with_suffix(".json.tmp")
    with scratch.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    scratch.replace(RECORD)


# --------------------------------------------------------------------------
# preconditions -- offline, no model
# --------------------------------------------------------------------------


def check_preconditions() -> tuple[bool, dict]:
    from pydantic import TypeAdapter

    from cerl.actions import Action
    from cerl.env.reward import W_COMMITTED_COST, W_DECISION, W_OUTCOME, W_TASK
    from cerl.reference.runner import run_actions
    from cerl.scenario import freeze
    from cerl_rl import protocol_v2

    adapter: TypeAdapter[Action] = TypeAdapter(Action)
    rows = [json.loads(line) for line in (TRIAL / "turns.jsonl").read_text().splitlines()]
    by_rollout: dict[int, list[dict]] = {}
    for row in rows:
        by_rollout.setdefault(row["rollout"], []).append(row)

    problems: list[str] = []
    episodes: dict[int, dict] = {}

    if sorted(by_rollout) != [0, 1, 2, 3]:
        problems.append(f"expected rollouts 0-3, found {sorted(by_rollout)}")

    for index in sorted(by_rollout):
        turns = sorted(by_rollout[index], key=lambda t: t["step_index"])

        # (a) every turn carries its exact provenance
        missing = [
            f"step {t['step_index']}: {field}"
            for t in turns
            for field in ("prompt_ids", "generated_ids", "action", "outcome")
            if not t.get(field)
        ]
        if missing:
            problems.append(f"rollout {index} missing provenance: {missing[:4]}")

        # (b) step indices are contiguous from zero -- no turn lost mid-episode
        if [t["step_index"] for t in turns] != list(range(len(turns))):
            problems.append(f"rollout {index} has non-contiguous step indices")

        # (c) the episode reached its ORIGINAL termination condition
        declared = next((t["action_kind"] for t in turns if t["action_kind"] in TERMINAL), None)
        at_limit = len(turns) >= protocol_v2.MAX_ACTIONS
        if declared is None and not at_limit:
            problems.append(
                f"rollout {index} ended after {len(turns)} turns with no terminal "
                f"declaration and below the {protocol_v2.MAX_ACTIONS}-action limit: "
                f"it was interrupted, not finished",
            )
        if declared is not None and turns[-1]["action_kind"] != declared:
            problems.append(f"rollout {index} continued past its terminal action")

        # (d) replay reproduces the outcomes and the reward
        scenario_id = turns[0]["scenario_id"]
        scenario = freeze.load(freeze.FROZEN_DIR / f"{scenario_id}.json")
        actions = tuple(adapter.validate_python(t["action"]) for t in turns)
        replayed = run_actions(scenario, actions)
        verdict = replayed.verdict
        entries = replayed.trace.agent_entries()
        if len(entries) != len(turns):
            problems.append(
                f"rollout {index}: replay executed {len(entries)} of {len(turns)} actions",
            )
        recorded_outcomes = [t["outcome"] for t in turns]
        replayed_outcomes = [str(e.outcome) for e in entries]
        if recorded_outcomes != replayed_outcomes:
            problems.append(f"rollout {index}: outcomes differ on replay")

        reward = (
            W_OUTCOME * (1.0 if verdict.correct_final_state else 0.0)
            + W_TASK * float(verdict.task_completion)
            + W_DECISION * (1.0 if verdict.decision_correct else 0.0)
            - W_COMMITTED_COST * len(verdict.violations)
        )
        episodes[index] = {
            "scenario_id": scenario_id, "turns": turns, "reward": reward,
            "task_completion": float(verdict.task_completion),
            "safe_completion": bool(verdict.safe_completion),
            "decision_correct": bool(verdict.decision_correct),
            "committed": [str(v.cost_class) for v in verdict.violations],
            "attempted": [str(v.cost_class) for v in verdict.attempted_violations],
            "failure_class": str(verdict.failure_class) if verdict.failure_class else None,
            "declared_outcome": declared,
            "termination": "declared:" + declared if declared else "action_limit",
            "tool_calls": int(verdict.tool_calls),
            "trace_head_hash": replayed.trace.head_hash,
            "terminal_state_hash": replayed.final.state_hash(),
            "generated_tokens": sum(len(t["generated_ids"]) for t in turns),
        }

    # cross-check against the trial's own replay record
    stored = json.loads((TRIAL / "replay.json").read_text())
    for row in stored["rollouts"]:
        mine = episodes.get(row["rollout"])
        if mine and abs(mine["reward"] - row["reward"]) > 1e-9:
            problems.append(f"rollout {row['rollout']}: reward differs from the trial's replay")

    summary = {
        "rollouts": len(episodes),
        "per_rollout": {
            str(index): {
                "turns": len(episode["turns"]),
                "reward": episode["reward"],
                "termination": episode["termination"],
                "task_completion": episode["task_completion"],
                "safe_completion": episode["safe_completion"],
                "decision_correct": episode["decision_correct"],
                "generated_tokens": episode["generated_tokens"],
            }
            for index, episode in sorted(episodes.items())
        },
        "problems": problems,
        "passed": not problems,
    }
    return not problems, {"summary": summary, "episodes": episodes}


# --------------------------------------------------------------------------
# the update itself
# --------------------------------------------------------------------------

#: Recorded so this recovery is reproducible. The original trial did **not**
#: record a LoRA initialisation seed, so its exact adapter cannot be recreated
#: -- see ``policy_equivalence`` in the report for what that does and does not
#: affect.
RECOVERY_SEED = 20260908


def build_episodes(episodes: dict[int, dict]):
    """Rebuild the group through the v2 data structures, from the record."""
    from cerl_rl.rollout import Episode
    from cerl_rl.rollout_v2 import EpisodeV2, TurnRecord

    built = []
    for index in sorted(episodes):
        source = episodes[index]
        turns = tuple(
            TurnRecord(
                step_index=turn["step_index"],
                prompt_ids=tuple(turn["prompt_ids"]),
                generated_ids=tuple(turn["generated_ids"]),
                stop_reason=turn["stop_reason"],
                text=turn.get("text", ""),
                action_kind=turn["action_kind"],
                action=turn["action"],
                category=turn["category"],
                outcome=turn["outcome"],
            )
            for turn in source["turns"]
        )
        from pydantic import TypeAdapter

        from cerl.actions import Action

        adapter: TypeAdapter[Action] = TypeAdapter(Action)
        episode = Episode(
            scenario_id=source["scenario_id"], turns=(),
            actions=tuple(adapter.validate_python(t["action"]) for t in source["turns"]),
            reward=source["reward"], task_completion=source["task_completion"],
            safe_completion=source["safe_completion"],
            decision_correct=source["decision_correct"],
            committed_violations=tuple(source["committed"]),
            attempted_violations=tuple(source["attempted"]),
            failure_class=source["failure_class"],
            declared_outcome=source["declared_outcome"],
            tool_calls=source["tool_calls"], malformed_actions=0,
            step_limited=source["declared_outcome"] is None,
            terminal_state_hash=source["terminal_state_hash"],
            trace_head_hash=source["trace_head_hash"],
        )
        built.append(EpisodeV2(turns=turns, episode=episode,
                               termination=source["termination"]))
    return built


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()

    ok, data = check_preconditions()
    record["preconditions"] = data["summary"]
    save()
    if not ok:
        record["status"] = "STOPPED: preconditions failed; the group is incomplete"
        save()
        return 2

    import torch

    from cerl_rl import protocol_v2
    from cerl_rl.grpo import advantages_for, turn_backward
    from cerl_rl.model import adapter_state, load_policy, pick_device

    record["config"] = {"learning_rate": args.lr, "lora_rank": args.lora_rank,
                        "objective": protocol_v2.REWARD_NAME, "kl_beta": 0.0,
                        "weight_decay": 0.0, "recovery_seed": RECOVERY_SEED}
    save()

    torch.manual_seed(RECOVERY_SEED)
    device = pick_device()
    model = load_policy(device, lora_rank=args.lora_rank)
    model.eval()
    record["load_seconds"] = round(time.monotonic() - started, 1)

    # -- policy equivalence ------------------------------------------------
    state = {name: p for name, p in model.named_parameters() if p.requires_grad}
    b_tensors = {n: p for n, p in state.items() if "lora_B" in n}
    a_tensors = {n: p for n, p in state.items() if "lora_A" in n}
    all_zero = all(float(p.detach().abs().sum()) == 0.0 for p in b_tensors.values())

    probe_ids = torch.tensor(
        data["episodes"][0]["turns"][0]["prompt_ids"], device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        with model.disable_adapter():
            base_logits = model(input_ids=probe_ids).logits[0, -1].float().cpu()
        tuned_logits = model(input_ids=probe_ids).logits[0, -1].float().cpu()
    identical = bool(torch.equal(base_logits, tuned_logits))

    record["policy_equivalence"] = {
        "lora_B_tensors": len(b_tensors), "lora_A_tensors": len(a_tensors),
        "all_lora_B_zero": all_zero,
        "adapter_enabled_logits_identical_to_disabled": identical,
        "probe_prompt_tokens": int(probe_ids.shape[1]),
        "argument": (
            "LoRA computes base + scaling * B @ A. With every B exactly zero the "
            "adapter contributes nothing, so the pre-update policy IS the pinned "
            "base model -- verified here by a bit-identical logit comparison, not "
            "assumed."
        ),
        "limitation": (
            "The original trial recorded no LoRA initialisation seed, so its exact "
            "A matrices cannot be recreated. That does not affect the pre-update "
            "policy, which is identical for any A while B is zero. It does affect "
            "the update: dL/dB depends on A, so the parameter change computed here "
            "is not bit-identical to the one that trial would have produced. This "
            "recovery is seeded and reproducible going forward."
        ),
    }
    save()
    if not all_zero or not identical:
        record["status"] = "STOPPED: the fresh policy is not equivalent to the base model"
        save()
        return 3

    # -- rebuild, advantage, backward -------------------------------------
    episodes = build_episodes(data["episodes"])
    rewards = [e.episode.reward for e in episodes]
    advantages = advantages_for(rewards)
    record["group"] = {
        "rewards": rewards,
        "distinct_rewards": len({round(r, 12) for r in rewards}),
        "advantages": advantages,
        "turns": sum(len(e.turns) for e in episodes),
        "generated_tokens": sum(e.generated_tokens for e in episodes),
    }
    save()
    if advantages is None:
        record["status"] = "SKIPPED: no within-group reward variation"
        save()
        return 0

    before = adapter_state(model)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)

    backward_started = time.monotonic()
    loss, tokens = turn_backward(model, episodes, advantages, device)
    record["backward"] = {"completed": True, "loss": loss, "scored_tokens": tokens,
                          "seconds": round(time.monotonic() - backward_started, 1)}
    save()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0
    finite = bool(torch.isfinite(torch.tensor(grad_norm)))
    record["gradient"] = {"norm": grad_norm, "finite": finite,
                          "tensors_with_grad": len(grads)}
    save()
    if not finite or grad_norm == 0.0:
        record["status"] = (
            f"SKIPPED: gradient is {'non-finite' if not finite else 'exactly zero'}; "
            f"the step would not be reward-driven"
        )
        record["update"] = {"occurred": False}
        save()
        return 0

    optimizer.step()
    after = adapter_state(model)
    delta = {k: (after[k] - before[k]).abs() for k in before}
    record["update"] = {
        "occurred": True,
        "optimizer": "AdamW", "learning_rate": args.lr,
        "l1": sum(float(d.sum()) for d in delta.values()),
        "linf": max(float(d.max()) for d in delta.values()),
        "tensors_changed": sum(1 for d in delta.values() if float(d.max()) > 0),
        "tensors_total": len(delta),
        "lora_B_changed": sum(1 for k, d in delta.items() if "lora_B" in k and float(d.max()) > 0),
        "lora_A_changed": sum(1 for k, d in delta.items() if "lora_A" in k and float(d.max()) > 0),
    }
    save()

    checkpoint = OUT / "adapter"
    model.save_pretrained(checkpoint)
    weights = next(checkpoint.glob("adapter_model*"), None)
    record["checkpoint"] = {
        "path": str(checkpoint), "weights_file": weights.name if weights else None,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None,
        "bytes": weights.stat().st_size if weights else None,
    }
    record["total_seconds"] = round(time.monotonic() - started, 1)
    record["status"] = "completed: one reward-driven optimizer update from recorded rollouts"
    save()
    print(json.dumps({k: record[k] for k in
                      ("status", "group", "gradient", "update", "checkpoint")}, indent=2))
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
