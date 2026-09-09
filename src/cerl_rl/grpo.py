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

from collections.abc import Callable
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


def release_cache() -> None:
    """Return cached accelerator memory, on whatever backend is present.

    Called unconditionally as ``torch.mps.empty_cache()`` before, which raises
    on any machine without the MPS backend -- so the CPU arithmetic tests, which
    are supposed to be the portable part of this work, failed to even run on
    Linux. The cache call is memory maintenance; it must never decide whether a
    correctness test can execute.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():  # pragma: no cover - no CUDA machine here
        torch.cuda.empty_cache()


def causal_lm(model: Any) -> Any:
    """The causal-LM module underneath whatever wrapper is in use.

    PEFT nests the real model at ``base_model.model``; a bare model is already
    it. Resolved through a function rather than hard-coded so the loss can be
    exercised on CPU against a small stand-in, which is how the equivalence
    tests in ``tests/rl/test_grpo_math.py`` run without downloading weights.
    """
    base = getattr(model, "base_model", None)
    return getattr(base, "model", model) if base is not None else model


def group_backward(
    model: Any,
    episodes: list[TokenisedEpisode],
    advantages: list[float],
    device: torch.device,
) -> tuple[float, int]:
    """Accumulate the group's gradient, one episode at a time.

    Two memory decisions, both forced by measurement rather than chosen.

    **One graph at a time.** Computing the whole group's loss and calling
    ``backward`` once kept four autograd graphs alive and hit ``MPS backend out
    of memory`` at 18.13 GiB. Backwarding per episode is arithmetically the same
    accumulated gradient with a quarter of the peak.

    **Logits only where the loss looks.** That alone was not enough. Qwen3's
    vocabulary is 151,936, so a full-sequence logit tensor for a 2,000-token
    episode is ~0.6 GB in bf16 and ~1.2 GB once ``cross_entropy`` upcasts it --
    and a second OOM landed inside ``cross_entropy`` itself. But the loss only
    ever reads the ~5% of positions the model generated. So the transformer is
    run for hidden states, and the language-model head is applied *only at the
    masked positions*. Hidden states are 1,024-wide and cost megabytes; the
    projection then runs over a couple of hundred rows instead of thousands.
    The gradient is unchanged -- the unmasked positions contributed zero to it
    anyway, which is precisely what the mask means.
    """
    total_loss = 0.0
    counted = 0
    inner = causal_lm(model)
    for tokenised, advantage in zip(episodes, advantages, strict=True):
        mask = tokenised.gen_mask
        if int(mask.sum()) == 0:
            continue
        ids = tokenised.input_ids.unsqueeze(0).to(device)

        # Predicting token t uses the hidden state at t-1, so a generated token
        # at position p is scored from position p-1.
        positions = (mask[1:] > 0).nonzero(as_tuple=True)[0].to(device)
        targets = ids[0, 1:][positions]

        hidden = inner.model(input_ids=ids).last_hidden_state[0]
        logits = inner.lm_head(hidden[positions]).float()
        logp = -torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        mean_logp = logp.mean()

        loss = -(advantage * mean_logp) / max(1, len(episodes))
        loss.backward()

        total_loss += float(loss.detach())
        counted += int(positions.numel())
        del hidden, logits, logp, mean_logp, loss
        release_cache()
    return total_loss, counted


def turn_backward(
    model: Any,
    episodes: list[Any],
    advantages: list[float],
    device: torch.device,
    on_turn_done: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, int]:
    """v2 loss: every turn scored under the exact prompt it was sampled with.

    The difference from :func:`group_backward` is not an optimisation, it is a
    correctness fix. v1 built one sequence per episode by re-rendering the whole
    conversation and locating each completion by text search, which could select
    tokens from the system prompt or an observation, and which scored earlier
    turns under a prefix the model never saw (Qwen's template drops the empty
    thinking block from earlier assistant messages when re-rendered). Here each
    turn is its own forward pass over ``prompt_ids + generated_ids`` -- the exact
    tokens recorded at generation -- and only that turn's generated positions
    are scored. Earlier turns appear as context and are never scored again.

    ``on_turn_done`` is called after each turn's backward, with that turn's
    running totals. It exists so a run stopped part way through can say exactly
    how far the computation reached: a backward over many long turns is the
    slowest thing here, and "it timed out somewhere inside" is not a useful
    record. It observes only -- it cannot change the gradient.

    **Weighting is unchanged and explicit.** The intent in v1 was: mean
    log-probability over an episode's generated tokens, then divided by group
    size, so every episode counts equally regardless of length. That is
    preserved by pooling an episode's turns -- summing log-probabilities over
    all its generated tokens and dividing by the episode's total -- rather than
    averaging per-turn means, which would have silently reweighted a short turn
    equal to a long one.
    """
    total_loss = 0.0
    counted = 0
    inner = causal_lm(model)
    for index, (tokenised, advantage) in enumerate(zip(episodes, advantages, strict=True)):
        turns = [t for t in tokenised.turns if t.generated_ids]
        if not turns:
            continue
        n_generated = sum(len(t.generated_ids) for t in turns)
        for turn in turns:
            sequence = torch.tensor(turn.sequence, device=device).unsqueeze(0)
            n_prompt = len(turn.prompt_ids)
            # Predicting token t uses the hidden state at t-1, so the first
            # generated token is scored from the last prompt position.
            positions = torch.arange(n_prompt - 1, sequence.shape[1] - 1, device=device)
            targets = sequence[0, n_prompt:]
            if positions.numel() != targets.numel():
                raise ValueError(
                    f"turn {turn.step_index}: {positions.numel()} scoring positions "
                    f"for {targets.numel()} generated tokens",
                )
            hidden = inner.model(input_ids=sequence).last_hidden_state[0]
            logits = inner.lm_head(hidden[positions]).float()
            logp = -torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            # Each turn contributes its share of the episode's token mean.
            loss = -(advantage * logp.sum() / n_generated) / max(1, len(episodes))
            loss.backward()
            total_loss += float(loss.detach())
            counted += int(targets.numel())
            if on_turn_done is not None:
                on_turn_done({
                    "episode": index, "step_index": turn.step_index,
                    "sequence_length": int(sequence.shape[1]),
                    "generated_tokens": int(targets.numel()),
                    "running_loss": total_loss, "running_tokens": counted,
                })
            del hidden, logits, logp, loss
            release_cache()
    return total_loss, counted
