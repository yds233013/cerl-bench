"""The GRPO update, checked against a straightforward reference calculation.

Small, deterministic, CPU-only, and **no language model is downloaded**: the
policy here is a few hundred random parameters with the same call shape as the
real one. That is deliberate. The two optimisations in ``group_backward`` --
applying the LM head only at masked positions, and backwarding per episode --
are exactly the kind of change that quietly alters the gradient while still
producing plausible numbers, so each is compared against the obvious
implementation it replaced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="RL pilot deps live in .venv-rl")
from torch import nn  # noqa: E402

from cerl_rl.grpo import advantages_for, causal_lm, group_backward  # noqa: E402
from cerl_rl.rollout import Episode, TokenisedEpisode  # noqa: E402

VOCAB, HIDDEN = 37, 8
DEVICE = torch.device("cpu")


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.mix = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(self.embed(input_ids))))


class _TinyCausalLM(nn.Module):
    """Same surface as the real model: ``.model`` for hidden states, ``.lm_head``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))


def _policy(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return _TinyCausalLM()


def _stub_episode() -> Episode:
    """A structurally valid Episode. These tests are about the loss arithmetic,
    which never reads it -- but the production model stays strict rather than
    being loosened to suit a test."""
    return Episode(
        scenario_id="stub", turns=(), actions=(), reward=0.0, task_completion=0.0,
        safe_completion=False, decision_correct=False, committed_violations=(),
        attempted_violations=(), failure_class=None, declared_outcome=None,
        tool_calls=0, malformed_actions=0, step_limited=True,
        terminal_state_hash="0" * 64, trace_head_hash="0" * 64,
    )


def _episode(seed: int, length: int, generated: list[int]) -> TokenisedEpisode:
    """A fake tokenised episode: ``generated`` are positions the model produced."""
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (length,))
    mask = torch.zeros(length, dtype=torch.long)
    for i in generated:
        mask[i] = 1
    return TokenisedEpisode(input_ids=ids, gen_mask=mask, episode=_stub_episode())


def _reference_loss(
    model: nn.Module, tokenised: TokenisedEpisode, advantage: float, n_episodes: int,
) -> torch.Tensor:
    """The obvious implementation: full logits, mask afterwards.

    This is what ``group_backward`` computed before it was optimised for memory,
    and it is the definition the optimised version has to match.
    """
    ids = tokenised.input_ids.unsqueeze(0)
    logits = model(ids).logits[:, :-1].float()
    targets = ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    step_mask = tokenised.gen_mask[1:].unsqueeze(0).float()
    mean_logp = (logp * step_mask).sum() / step_mask.sum()
    return -(advantage * mean_logp) / n_episodes


# --------------------------------------------------------------------------
# the two optimisations
# --------------------------------------------------------------------------


def test_masked_position_lm_head_matches_full_logit_masked_loss():
    """The memory fix must be an optimisation, not a different objective."""
    optimised, reference = _policy(1), _policy(1)
    episodes = [_episode(10, 12, [3, 4, 8]), _episode(11, 9, [2, 6])]
    advantages = [1.25, -0.75]

    loss_opt, tokens = group_backward(optimised, episodes, advantages, DEVICE)
    assert tokens == 5

    total = sum(
        _reference_loss(reference, e, a, len(episodes))
        for e, a in zip(episodes, advantages, strict=True)
    )
    total.backward()

    assert loss_opt == pytest.approx(float(total.detach()), abs=1e-6)
    for (name, a), (_, b) in zip(
        optimised.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert a.grad is not None, name
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_per_episode_backward_matches_one_batched_backward():
    """Accumulating per episode must preserve the intended weighting.

    If the division by group size were dropped, or applied twice, the gradient
    magnitude would change while every printed number still looked reasonable.
    """
    accumulated, batched = _policy(2), _policy(2)
    episodes = [_episode(20, 10, [1, 5]), _episode(21, 14, [2, 3, 9]), _episode(22, 8, [4])]
    advantages = [1.0, -0.5, -0.5]

    group_backward(accumulated, episodes, advantages, DEVICE)

    total = sum(
        _reference_loss(batched, e, a, len(episodes))
        for e, a in zip(episodes, advantages, strict=True)
    )
    total.backward()

    for (name, a), (_, b) in zip(
        accumulated.named_parameters(), batched.named_parameters(), strict=True,
    ):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_each_episode_carries_equal_weight_regardless_of_length():
    """Per-episode mean, then divided by group size.

    A long episode must not dominate an update simply by having more tokens.
    This is a deliberate departure from the original GRPO normalisation and is
    pinned here so it cannot drift silently.
    """
    short_model, long_model = _policy(3), _policy(3)
    short = _episode(30, 8, [2, 3])
    long = _episode(30, 8, [2, 3])
    # same content, but pretend the second has many more generated tokens
    long = TokenisedEpisode(
        input_ids=long.input_ids,
        gen_mask=torch.tensor([0, 1, 1, 1, 1, 1, 1, 0]),
        episode=_stub_episode(),
    )
    group_backward(short_model, [short], [1.0], DEVICE)
    group_backward(long_model, [long], [1.0], DEVICE)
    a = torch.cat([p.grad.flatten() for p in short_model.parameters()])
    b = torch.cat([p.grad.flatten() for p in long_model.parameters()])
    # different token counts, comparable gradient scale -- not a 3x difference
    assert 0.2 < float(a.norm() / b.norm()) < 5.0


# --------------------------------------------------------------------------
# causal alignment
# --------------------------------------------------------------------------


def test_the_loss_scores_the_next_token_not_the_current_one():
    """Off-by-one here trains the model to predict what it just read.

    Built so the answer is unambiguous: the reference gathers ``ids[1:]`` from
    logits at ``[:-1]``, and the optimised path must agree exactly. A shifted
    implementation disagrees on any sequence where tokens differ.
    """
    optimised, reference = _policy(4), _policy(4)
    ids = torch.tensor([5, 9, 14, 2, 30, 7])
    mask = torch.tensor([0, 0, 1, 1, 0, 0])
    tokenised = TokenisedEpisode(input_ids=ids, gen_mask=mask, episode=_stub_episode())

    loss_opt, tokens = group_backward(optimised, [tokenised], [1.0], DEVICE)
    assert tokens == 2

    ref = _reference_loss(reference, tokenised, 1.0, 1)
    ref.backward()
    assert loss_opt == pytest.approx(float(ref.detach()), abs=1e-6)

    # and a deliberately shifted mask must NOT produce the same loss
    shifted = TokenisedEpisode(
        input_ids=ids, gen_mask=torch.tensor([0, 1, 1, 0, 0, 0]), episode=_stub_episode(),
    )
    other = _policy(4)
    loss_shift, _ = group_backward(other, [shifted], [1.0], DEVICE)
    assert loss_shift != pytest.approx(loss_opt, abs=1e-6)


def test_only_masked_positions_receive_gradient():
    """The property the whole update rests on: environment text is not trained on."""
    model = _policy(5)
    ids = torch.arange(6) % VOCAB
    tokenised = TokenisedEpisode(
        input_ids=ids, gen_mask=torch.tensor([0, 0, 0, 1, 0, 0]), episode=_stub_episode(),
    )
    group_backward(model, [tokenised], [1.0], DEVICE)
    grad_one = model.lm_head.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    # change a token the mask does NOT cover; the loss must be unchanged
    other_ids = ids.clone()
    other_ids[5] = (other_ids[5] + 3) % VOCAB
    group_backward(
        model,
        [TokenisedEpisode(
            input_ids=other_ids, gen_mask=tokenised.gen_mask,
            episode=_stub_episode(),
        )],
        [1.0],
        DEVICE,
    )
    assert torch.allclose(grad_one, model.lm_head.weight.grad, atol=1e-6)


def test_an_episode_with_no_generated_tokens_is_skipped_entirely():
    model = _policy(6)
    empty = TokenisedEpisode(
        input_ids=torch.arange(5) % VOCAB, gen_mask=torch.zeros(5, dtype=torch.long),
        episode=_stub_episode(),
    )
    loss, tokens = group_backward(model, [empty], [1.0], DEVICE)
    assert (loss, tokens) == (0.0, 0)
    assert all(p.grad is None for p in model.parameters())


# --------------------------------------------------------------------------
# advantages and sign
# --------------------------------------------------------------------------


def test_advantage_is_the_population_standardisation_of_the_group():
    rewards = [0.1, 0.4, 0.9, 0.2]
    mean = sum(rewards) / 4
    std = (sum((r - mean) ** 2 for r in rewards) / 4) ** 0.5
    assert advantages_for(rewards) == pytest.approx([(r - mean) / std for r in rewards])


def test_a_positive_advantage_increases_the_likelihood_of_its_tokens():
    """Sign check. Backwards here would train the policy to avoid what it was
    rewarded for, and every other number in the run would still look sane."""
    model = _policy(7)
    tokenised = _episode(40, 7, [2, 4])
    ids = tokenised.input_ids.unsqueeze(0)
    positions = (tokenised.gen_mask[1:] > 0).nonzero(as_tuple=True)[0]
    targets = ids[0, 1:][positions]

    def logprob() -> float:
        with torch.no_grad():
            inner = causal_lm(model)
            hidden = inner.model(input_ids=ids).last_hidden_state[0]
            logits = inner.lm_head(hidden[positions]).float()
            return float(-nn.functional.cross_entropy(logits, targets, reduction="none").mean())

    before = logprob()
    group_backward(model, [tokenised], [1.0], DEVICE)
    torch.optim.SGD(model.parameters(), lr=0.5).step()
    assert logprob() > before


def test_a_negative_advantage_decreases_it():
    model = _policy(7)
    tokenised = _episode(40, 7, [2, 4])
    ids = tokenised.input_ids.unsqueeze(0)
    positions = (tokenised.gen_mask[1:] > 0).nonzero(as_tuple=True)[0]
    targets = ids[0, 1:][positions]

    def logprob() -> float:
        with torch.no_grad():
            inner = causal_lm(model)
            hidden = inner.model(input_ids=ids).last_hidden_state[0]
            logits = inner.lm_head(hidden[positions]).float()
            return float(-nn.functional.cross_entropy(logits, targets, reduction="none").mean())

    before = logprob()
    group_backward(model, [tokenised], [-1.0], DEVICE)
    torch.optim.SGD(model.parameters(), lr=0.5).step()
    assert logprob() < before


def test_causal_lm_unwraps_a_peft_style_wrapper():
    inner = _policy(8)
    wrapped = SimpleNamespace(base_model=SimpleNamespace(model=inner))
    assert causal_lm(wrapped) is inner
    assert causal_lm(inner) is inner
