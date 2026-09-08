"""A minimal, explicit GRPO update.

**Why this is not ``trl.GRPOTrainer``.** It was investigated first. In TRL
0.24.0 ``GRPOTrainer`` is single-turn: ``_generate(prompts, images)`` produces
one completion per prompt, which is then scored by a reward function. There is
no ``environment_factory`` -- the string does not appear anywhere in the
package -- and no hook for stepping an environment between turns. Two
requirements of this pilot cannot be expressed through it:

* a rollout is a *multi-turn* conversation in which the environment produces
  every other message, and
* the loss must cover only the model's own tokens, while TRL treats the whole
  completion as trainable.

Bending a private method (``_generate_and_score_completions``) into that shape
would be more code than this file, and less legible. GRPO itself is small: for
each group of rollouts on one scenario, advantage is the reward standardised
within the group, and the loss is the negative advantage-weighted log-likelihood
of the tokens the policy chose. That is what this implements, and nothing else.
"""

from __future__ import annotations

from typing import Any

import torch

from cerl_rl.rollout import TokenisedEpisode


class GroupStats:
    """What one optimizer update actually saw. Recorded, not inferred."""

    __slots__ = (
        "advantages", "grad_norm", "loss", "reward_mean", "reward_std",
        "rewards", "scenario_id", "skip_reason", "skipped", "tokens",
    )

    def __init__(self, scenario_id: str, rewards: list[float]) -> None:
        self.scenario_id = scenario_id
        self.rewards = rewards
        self.reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        var = (
            sum((r - self.reward_mean) ** 2 for r in rewards) / len(rewards)
            if rewards else 0.0
        )
        self.reward_std = var ** 0.5
        self.advantages: list[float] = []
        self.loss: float | None = None
        self.grad_norm: float | None = None
        self.tokens = 0
        self.skipped = False
        self.skip_reason = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "rewards": self.rewards,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "advantages": self.advantages,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "generated_tokens": self.tokens,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


#: Two rollouts is the minimum that can have a spread; one has nothing to be
#: relative to, and GRPO is group-*relative*.
_MIN_GROUP = 2


def advantages_for(rewards: list[float], *, eps: float = 1e-4) -> list[float] | None:
    """Group-relative advantage, or ``None`` when the group carries no signal.

    If every rollout in a group earned the same reward the advantages are all
    zero, the gradient is zero, and an optimizer step would move the parameters
    only through Adam's state -- not through the reward. Returning ``None`` makes
    that case visible and skippable, so "20 updates" cannot quietly mean "20
    updates on no signal". This distinction is the whole difference between
    reward-driven training and a loop that merely ran.
    """
    n = len(rewards)
    if n < _MIN_GROUP:
        return None
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return None
    return [(r - mean) / std for r in rewards]


def group_backward(
    model: Any,
    episodes: list[TokenisedEpisode],
    advantages: list[float],
    device: torch.device,
) -> tuple[float, int]:
    """Accumulate the group's gradient, one episode at a time.

    Written this way for a measured reason. Computing the whole group's loss and
    calling ``backward`` once kept four autograd graphs alive at the same time
    and hit ``MPS backend out of memory`` at 18.13 GiB. The cause is the vocabulary:
    Qwen3 has 151,936 entries, so the logits for a ~900-token episode are ~270 MB
    in bf16 and ~550 MB once upcast, and a separate ``log_softmax`` buffer doubles
    it again. Four of those, retained for backward, do not fit on a 16 GB machine
    that is already in swap.

    Two changes fix it and neither weakens the update. Each episode is backwarded
    immediately so only one graph exists at a time -- the accumulated gradient is
    arithmetically identical to the batched one. And the per-token log-probability
    comes from ``cross_entropy``, which computes it without materialising a full
    log-softmax tensor of its own.
    """
    total_loss = 0.0
    counted = 0
    for tokenised, advantage in zip(episodes, advantages, strict=True):
        mask = tokenised.gen_mask
        if int(mask.sum()) == 0:
            continue
        ids = tokenised.input_ids.unsqueeze(0).to(device)
        step_mask = mask[1:].to(device).float()

        logits = model(input_ids=ids).logits[:, :-1]
        targets = ids[:, 1:]
        # -cross_entropy is the log-probability of the realised token, computed
        # without a second vocabulary-sized buffer.
        logp = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="none",
        )
        mean_logp = (logp * step_mask).sum() / step_mask.sum()
        loss = -(advantage * mean_logp) / max(1, len(episodes))
        loss.backward()

        total_loss += float(loss.detach())
        counted += int(step_mask.sum().item())
        del logits, logp, loss
        torch.mps.empty_cache()
    return total_loss, counted
