"""The runnable **v2** pilot.

Separate from :mod:`cerl_rl.pilot`, which is left exactly as it ran so the
recorded experiment stays reproducible. This one uses the corrected path
throughout: :mod:`cerl_rl.protocol_v2` for limits and sampling,
:func:`cerl_rl.rollout_v2.rollout_v2` for token-exact rollouts, and
:func:`cerl_rl.grpo.turn_backward` for a loss that scores each turn under the
prompt it was actually generated with. It never calls the v1 rollout or
``group_backward``; ``tests/rl/test_pilot_v2.py`` asserts that mechanically.

Two structural choices worth stating.

**The model is injected, not constructed.** ``run_pilot`` takes the model and
tokenizer as arguments, so the whole loop -- rollouts, advantages, backward,
records, deadline behaviour -- can be exercised with fakes and no weights.
``main`` is the thin part that loads a real model.

**Token provenance is written to its own append-only file.** Every turn's exact
prompt and generated ids go to ``turns.jsonl``, one JSON object per line, as the
run proceeds. Keeping them out of ``pilot_run.json`` keeps that file readable,
and appending as we go means an interrupted run still has the provenance for the
turns it completed -- which is the evidence a later loss audit needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from typing import Any

import torch

from cerl.agents.prompt_only import SYSTEM_PROMPT
from cerl_rl import protocol_v2
from cerl_rl.environment import ActionCategory
from cerl_rl.grpo import GroupStats, advantages_for, release_cache, turn_backward
from cerl_rl.rollout_v2 import EpisodeV2, rollout_v2, system_prompt_v2
from cerl_rl.runtime import Deadline, DeadlineExceeded, prepare_run_directory, write_atomic

DEFAULT_OUT = pathlib.Path("evidence/rl-pilot-v2")


class TurnLog:
    """Append-only record of exact per-turn tokens."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.turns = 0

    def write_turn(self, turn: Any, *, phase: str, update: int | None,
                   rollout: int, scenario_id: str) -> None:
        """Append one completed turn, flushed to disk immediately.

        Called from the rollout as each turn finishes rather than after the
        episode, so an interrupted episode still leaves the turns it really
        executed. Those turns were genuinely produced by the environment;
        discarding them because a later turn overran would throw away evidence.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "phase": phase,
                        "update": update,
                        "rollout": rollout,
                        "scenario_id": scenario_id,
                        "step_index": turn.step_index,
                        "prompt_ids": list(turn.prompt_ids),
                        "generated_ids": list(turn.generated_ids),
                        "stop_reason": turn.stop_reason,
                        "action_kind": turn.action_kind,
                        "action": turn.action,
                        "category": turn.category,
                        "outcome": turn.outcome,
                    },
                )
                + "\n",
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.turns += 1


def _save_checkpoint(model: Any, out: pathlib.Path, *, update: int | None) -> dict[str, Any]:
    """Write the adapter and describe it. Called after every completed update."""
    path = out / "adapter"
    model.save_pretrained(path)
    weights = next(path.glob("adapter_model*"), None)
    return {
        "path": str(path),
        "after_update": update,
        "weights_file": weights.name if weights else None,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None,
    }


def episode_row(episode: EpisodeV2) -> dict[str, Any]:
    """One episode, summarised. Token ids live in ``turns.jsonl``."""
    inner = episode.episode
    return {
        "scenario_id": inner.scenario_id,
        "reward": inner.reward,
        "task_completion": inner.task_completion,
        "safe_completion": inner.safe_completion,
        "decision_correct": inner.decision_correct,
        "declared_outcome": inner.declared_outcome,
        "failure_class": inner.failure_class,
        "committed": list(inner.committed_violations),
        "attempted": list(inner.attempted_violations),
        # R6: the verifier's definition, with the other categories kept apart.
        "tool_calls": inner.tool_calls,
        "terminal_declarations": sum(
            1 for t in episode.turns if t.category == ActionCategory.TERMINAL.value
        ),
        "malformed_actions": inner.malformed_actions,
        "total_actions": len(episode.turns),
        # R9: context exhaustion and action exhaustion are different events.
        "termination": episode.termination,
        "stop_reasons": [t.stop_reason for t in episode.turns],
        "generated_tokens": episode.generated_tokens,
        "actions": [a.model_dump(mode="json") for a in inner.actions],
        "terminal_state_hash": inner.terminal_state_hash,
        "trace_head_hash": inner.trace_head_hash,
    }


def _aggregate(rows: list[dict[str, Any]], label: str, seconds: float) -> dict[str, Any]:
    n = len(rows)
    return {
        "label": label,
        "seconds": round(seconds, 1),
        "episodes": n,
        "reward_mean": sum(r["reward"] for r in rows) / n if n else 0.0,
        "task_completion_mean": sum(r["task_completion"] for r in rows) / n if n else 0.0,
        "safe_completion": sum(1 for r in rows if r["safe_completion"]),
        "decision_correct": sum(1 for r in rows if r["decision_correct"]),
        "committed_violations": sum(len(r["committed"]) for r in rows),
        "attempted_violations": sum(len(r["attempted"]) for r in rows),
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "terminal_declarations": sum(r["terminal_declarations"] for r in rows),
        "malformed_actions": sum(r["malformed_actions"] for r in rows),
        "generated_tokens": sum(r["generated_tokens"] for r in rows),
        "terminations": {
            reason: sum(1 for r in rows if r["termination"] == reason)
            for reason in sorted({r["termination"] for r in rows})
        },
        "per_episode": rows,
    }


def evaluate_v2(
    model: Any,
    tokenizer: Any,
    scenarios: list[Any],
    device: Any,
    *,
    label: str,
    deadline: Deadline,
    turn_log: TurnLog,
    system: str,
) -> dict[str, Any]:
    """The before/after measurement, identical in both phases."""
    model.eval()
    torch.manual_seed(protocol_v2.EVAL_SEED)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, scenario in enumerate(scenarios):
        deadline.enforce(f"{label} evaluation, scenario {index}")
        episode = rollout_v2(
            model, tokenizer, scenario,
            system_prompt=system,
            max_actions=protocol_v2.MAX_ACTIONS,
            max_new_tokens=protocol_v2.MAX_NEW_TOKENS,
            max_prompt_tokens=protocol_v2.MAX_PROMPT_TOKENS,
            temperature=protocol_v2.EVAL_TEMPERATURE,
            device=device,
            before_generate=lambda step: deadline.enforce(f"{label} generation, step {step}"),
            on_turn=lambda turn, i=index, sid=scenario.scenario_id: turn_log.write_turn(
                turn, phase=label, update=None, rollout=i, scenario_id=sid,
            ),
        )
        rows.append(episode_row(episode))
        release_cache()
    return _aggregate(rows, label, time.monotonic() - started)


def run_pilot(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    out: pathlib.Path,
    updates: int,
    group_size: int,
    learning_rate: float,
    deadline: Deadline,
    train_scenarios: list[Any],
    val_scenarios: list[Any],
    reserve_seconds: float = 8 * 60,
    system: str | None = None,
    skip_evaluation: bool = False,
) -> dict[str, Any]:
    """Baseline, GRPO training, and the same evaluation again -- all v2.

    ``skip_evaluation`` runs the training loop alone. It exists for a bounded
    trial whose question is only whether the training path works end to end on
    real weights; running a before/after measurement there would spend most of
    the budget producing numbers nobody asked for, and would touch validation
    scenarios the trial has no business touching.
    """
    from cerl_rl.model import adapter_state

    system = system or system_prompt_v2(SYSTEM_PROMPT)
    turn_log = TurnLog(out / "turns.jsonl")
    record: dict[str, Any] = {
        "protocol_version": protocol_v2.PROTOCOL_VERSION,
        "uses": {
            "protocol": "cerl_rl.protocol_v2",
            "rollout": "cerl_rl.rollout_v2.rollout_v2",
            "loss": "cerl_rl.grpo.turn_backward",
            "tool_contract": "complete public schemas (cerl_rl.tools.tool_contract)",
        },
        "limits": {
            "max_actions": protocol_v2.MAX_ACTIONS,
            "max_prompt_tokens": protocol_v2.MAX_PROMPT_TOKENS,
            "max_new_tokens": protocol_v2.MAX_NEW_TOKENS,
        },
        "sampling": {
            "train_temperature": protocol_v2.TRAIN_TEMPERATURE,
            "eval_temperature": protocol_v2.EVAL_TEMPERATURE,
            **dict(protocol_v2.SAMPLING_NEUTRALISED),
        },
        "config": {
            "updates_requested": updates,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "kl_beta": 0.0,
            "deadline_seconds": deadline.limit,
        },
        "groups": [],
        "reward_driven_updates": 0,
        "interruption": None,
        "stopped_because": None,
    }

    def save() -> None:
        record["turns_recorded"] = turn_log.turns
        write_atomic(out / "pilot_run.json", record)

    def interrupted(phase: str, error: Exception) -> None:
        """Record the interruption *before* attempting anything else."""
        record["interruption"] = {"phase": phase, "detail": str(error),
                                  "elapsed_seconds": round(deadline.elapsed, 1)}
        record["stopped_because"] = f"interrupted during {phase}"
        save()

    record["skip_evaluation"] = skip_evaluation
    save()
    try:
        if skip_evaluation:
            record["baseline"] = {"label": "baseline", "skipped": "training-only trial"}
        else:
            record["baseline"] = evaluate_v2(
                model, tokenizer, val_scenarios, device,
                label="baseline", deadline=deadline, turn_log=turn_log, system=system,
            )
    except DeadlineExceeded as error:
        interrupted("baseline evaluation", error)
        return record
    save()

    initial = adapter_state(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=0.0,
    )
    reward_driven = 0

    for update in range(updates):
        scenario = train_scenarios[update % len(train_scenarios)]
        try:
            deadline.enforce(f"training update {update}", reserve=reserve_seconds)
        except DeadlineExceeded as error:
            record["stopped_because"] = f"deadline reached after {update} updates"
            record["interruption"] = {"phase": f"training update {update}",
                                      "detail": str(error),
                                      "elapsed_seconds": round(deadline.elapsed, 1)}
            save()
            break

        model.eval()
        torch.manual_seed(protocol_v2.EVAL_SEED + update)
        episodes: list[EpisodeV2] = []
        generation_started = time.monotonic()
        try:
            for rollout_index in range(group_size):
                episodes.append(
                    rollout_v2(
                        model, tokenizer, scenario,
                        system_prompt=system,
                        max_actions=protocol_v2.MAX_ACTIONS,
                        max_new_tokens=protocol_v2.MAX_NEW_TOKENS,
                        max_prompt_tokens=protocol_v2.MAX_PROMPT_TOKENS,
                        temperature=protocol_v2.TRAIN_TEMPERATURE,
                        device=device,
                        before_generate=lambda step, u=update: deadline.enforce(
                            f"update {u} generation, step {step}",
                        ),
                        on_turn=lambda turn, u=update, r=rollout_index,
                        sid=scenario.scenario_id: turn_log.write_turn(
                            turn, phase="train", update=u, rollout=r, scenario_id=sid,
                        ),
                    ),
                )
                release_cache()
        except DeadlineExceeded as error:
            interrupted(f"training update {update} generation", error)
            return record

        rewards = [e.episode.reward for e in episodes]
        stats = GroupStats(scenario.scenario_id, rewards)
        advantages = advantages_for(rewards)
        entry: dict[str, Any] = {
            "update": update,
            "scenario_id": scenario.scenario_id,
            "generation_seconds": round(time.monotonic() - generation_started, 1),
            "episodes": [episode_row(e) for e in episodes],
        }

        if advantages is None:
            stats.skipped = True
            stats.skip_reason = "no within-group reward variation; advantages would be zero"
            entry.update(stats.as_dict())
            record["groups"].append(entry)
            save()
            continue

        stats.advantages = advantages
        model.train()
        optimizer.zero_grad(set_to_none=True)
        backward_started = time.monotonic()
        try:
            deadline.enforce(f"update {update} backward")
            loss, tokens = turn_backward(model, episodes, advantages, device)
        except DeadlineExceeded as error:
            interrupted(f"training update {update} backward", error)
            return record

        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0
        if not torch.isfinite(torch.tensor(grad_norm)):
            record["stopped_because"] = f"non-finite gradient at update {update}"
            save()
            break

        stats.loss, stats.grad_norm, stats.tokens = loss, grad_norm, tokens
        if grad_norm == 0.0:
            # A zero gradient moves parameters only through the optimizer's own
            # state, exactly like the zero-advantage groups the skip logic keeps
            # out of the count. Counting it as reward-driven would let "N
            # updates" mean something weaker than it says.
            stats.skipped = True
            stats.skip_reason = (
                "gradient was exactly zero; the step would not be reward-driven"
            )
            entry.update(stats.as_dict())
            record["groups"].append(entry)
            save()
            continue

        # R3-adjacent: the budget is checked immediately before the parameter
        # update, so an overrun cannot leave a half-applied optimizer state.
        try:
            deadline.enforce(f"update {update} optimizer step")
        except DeadlineExceeded as error:
            interrupted(f"training update {update} optimizer step", error)
            return record
        optimizer.step()
        reward_driven += 1

        entry.update(stats.as_dict())
        entry["backward_seconds"] = round(time.monotonic() - backward_started, 1)
        record["groups"].append(entry)
        record["reward_driven_updates"] = reward_driven
        # Checkpoint after every completed update. Saving only at the end meant
        # an interruption during training discarded every update that had
        # already been applied.
        record["checkpoint"] = _save_checkpoint(model, out, update=update)
        save()

    record["stopped_because"] = record["stopped_because"] or "completed all requested updates"

    final = adapter_state(model)
    delta = {k: (final[k] - initial[k]).abs() for k in initial}
    record["adapter_change"] = {
        "l1": sum(float(d.sum()) for d in delta.values()),
        "linf": max((float(d.max()) for d in delta.values()), default=0.0),
        "tensors_changed": sum(1 for d in delta.values() if float(d.max()) > 0),
        "tensors_total": len(delta),
    }

    record["checkpoint"] = _save_checkpoint(model, out, update=None)
    save()

    if skip_evaluation:
        record["after"] = {"label": "after", "skipped": "training-only trial"}
        record["total_seconds"] = round(deadline.elapsed, 1)
        save()
        return record

    try:
        record["after"] = evaluate_v2(
            model, tokenizer, val_scenarios, device,
            label="after", deadline=deadline, turn_log=turn_log, system=system,
        )
    except DeadlineExceeded as error:
        interrupted("after evaluation", error)
        return record

    record["total_seconds"] = round(deadline.elapsed, 1)
    save()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="CERL-Bench local RL pilot, protocol v2")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--deadline-minutes", type=float, default=85.0)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument(
        "--training-only", action="store_true",
        help="run the training loop alone; no validation scenarios are touched",
    )
    args = parser.parse_args()

    from cerl_rl.model import load_policy, load_tokenizer, pick_device

    out = prepare_run_directory(args.out, allow_existing=args.allow_existing)
    # Started BEFORE the model is loaded: loading is model work and counts
    # against the budget. Constructing it as an argument to run_pilot would have
    # started it after load_policy() had already returned.
    deadline = Deadline(args.deadline_minutes * 60)
    device = pick_device()
    policy = load_policy(device, lora_rank=args.lora_rank)
    tokenizer = load_tokenizer()
    deadline.enforce("model loading")
    record = run_pilot(
        policy,
        tokenizer,
        device,
        out=out,
        updates=args.updates,
        group_size=args.group_size,
        learning_rate=args.lr,
        deadline=deadline,
        train_scenarios=protocol_v2.training_selection().load(),
        # A training-only trial must not even load validation scenarios.
        val_scenarios=[] if args.training_only else protocol_v2.validation_selection().load(),
        skip_evaluation=args.training_only,
    )
    print(json.dumps({k: record.get(k) for k in
                      ("reward_driven_updates", "stopped_because", "interruption")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
