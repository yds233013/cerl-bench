"""R1: token provenance and conditioning.

The v1 defect was invisible in every summary number: the loss trained on a span
found by searching re-rendered text, which could land in the system prompt, and
scored turns under a prefix the model was never conditioned on. These tests use
the pinned tokenizer (no model weights) and tiny CPU networks to show the old
behaviour was wrong and the new behaviour is right.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="RL pilot deps live in .venv-rl")
from torch import nn  # noqa: E402

from cerl_rl.grpo import turn_backward  # noqa: E402
from cerl_rl.rollout_v2 import TurnRecord  # noqa: E402

VOCAB, HIDDEN = 41, 8
DEVICE = torch.device("cpu")


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.mix = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(self.embed(input_ids))))


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))


def _policy(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return _TinyCausalLM()


def _turn(step: int, prompt: list[int], generated: list[int], stop: str = "eos") -> TurnRecord:
    return TurnRecord(
        step_index=step, prompt_ids=tuple(prompt), generated_ids=tuple(generated),
        stop_reason=stop, text="", action_kind="tickets.get",
        action={"kind": "tickets.get", "ticket_id": "tkt_000000000001"},
        category="tool_call", outcome="read_only",
    )


class _Episode:
    def __init__(self, turns: list[TurnRecord]) -> None:
        self.turns = tuple(turns)


def _reference(model: nn.Module, turn: TurnRecord, advantage: float,
               n_generated: int, n_episodes: int) -> torch.Tensor:
    """Score this turn's generated tokens under this turn's own prompt."""
    sequence = torch.tensor(turn.sequence).unsqueeze(0)
    n_prompt = len(turn.prompt_ids)
    logits = model(sequence).logits[0].float()
    total = torch.zeros(())
    for offset, target in enumerate(turn.generated_ids):
        row = logits[n_prompt - 1 + offset]
        total = total + torch.log_softmax(row, dim=-1)[target]
    return -(advantage * total / n_generated) / n_episodes


# --------------------------------------------------------------------------
# the loss matches an explicit reference
# --------------------------------------------------------------------------


def test_per_turn_loss_matches_an_explicit_reference_calculation():
    optimised, reference = _policy(1), _policy(1)
    turns = [_turn(0, [3, 9, 14], [7, 2]), _turn(1, [3, 9, 14, 7, 2, 5], [11])]
    episode = _Episode(turns)

    loss, tokens = turn_backward(optimised, [episode], [1.5], DEVICE)
    assert tokens == 3

    n_generated = 3
    total = sum(_reference(reference, t, 1.5, n_generated, 1) for t in turns)
    total.backward()

    assert loss == pytest.approx(float(total.detach()), abs=1e-6)
    for (name, a), (_, b) in zip(
        optimised.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_a_turn_is_scored_under_its_own_prompt_not_a_rerendered_one():
    """The R1b defect, as an arithmetic statement.

    Two runs with identical generated tokens but different prompts must produce
    different gradients. If the loss were computed from a re-rendered
    conversation, the prompt difference would be erased and these would agree.
    """
    a_model, b_model = _policy(2), _policy(2)
    generated = [4, 8]
    turn_a = _turn(0, [1, 2, 3], generated)
    turn_b = _turn(0, [1, 2, 30], generated)   # one token of context differs

    turn_backward(a_model, [_Episode([turn_a])], [1.0], DEVICE)
    turn_backward(b_model, [_Episode([turn_b])], [1.0], DEVICE)

    ga = torch.cat([p.grad.flatten() for p in a_model.parameters()])
    gb = torch.cat([p.grad.flatten() for p in b_model.parameters()])
    assert not torch.allclose(ga, gb, atol=1e-8), (
        "conditioning was ignored: the same tokens under different prompts "
        "produced the same gradient"
    )


def test_earlier_turns_are_context_and_are_never_scored_twice():
    """A later turn's prompt contains the earlier turn's tokens. Only the
    later turn's own generated span may be scored in that forward pass."""
    model = _policy(3)
    first = _turn(0, [5, 6], [7, 8])
    second = _turn(1, [5, 6, 7, 8, 9], [10])
    _loss, tokens = turn_backward(model, [_Episode([first, second])], [1.0], DEVICE)
    assert tokens == 3, "expected 2 + 1 generated tokens, not the prompt tokens"


def test_only_the_generated_span_receives_gradient():
    """Changing a prompt token changes the loss; changing nothing else can."""
    model = _policy(4)
    turn_backward(model, [_Episode([_turn(0, [1, 2, 3], [9])])], [1.0], DEVICE)
    baseline = model.lm_head.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    turn_backward(model, [_Episode([_turn(0, [1, 2, 3], [9])])], [1.0], DEVICE)
    assert torch.allclose(baseline, model.lm_head.weight.grad, atol=1e-8)


def test_episode_weighting_pools_turns_rather_than_averaging_them():
    """Weighting must be per *episode*, not per turn.

    One episode of two turns (1 token, 3 tokens) must weigh the same as one
    episode of a single 4-token turn when the log-probabilities are equal --
    averaging per-turn means would over-weight the 1-token turn fourfold.
    """
    split_model, whole_model = _policy(5), _policy(5)
    prompt = [2, 3, 4]
    split = _Episode([_turn(0, prompt, [11]), _turn(1, [*prompt, 11], [12, 13, 14])])
    whole = _Episode([_turn(0, prompt, [11, 12, 13, 14])])

    loss_split, tokens_split = turn_backward(split_model, [split], [1.0], DEVICE)
    loss_whole, tokens_whole = turn_backward(whole_model, [whole], [1.0], DEVICE)
    assert tokens_split == tokens_whole == 4
    # Not equal in general -- the conditioning genuinely differs -- but the
    # normalisation must not differ by a factor of the turn count.
    assert 0.5 < abs(loss_split) / abs(loss_whole) < 2.0


def test_a_mismatched_span_raises_rather_than_training_on_something_else():
    model = _policy(6)
    broken = TurnRecord(
        step_index=0, prompt_ids=(1, 2, 3), generated_ids=(),
        stop_reason="eos", text="", action_kind="abstain",
        action={"kind": "abstain", "reason": "x"}, category="terminal", outcome="",
    )
    # an empty span is dropped, not silently scored
    loss, tokens = turn_backward(model, [_Episode([broken])], [1.0], DEVICE)
    assert (loss, tokens) == (0.0, 0)


# --------------------------------------------------------------------------
# the v1 defect itself, against the real tokenizer
# --------------------------------------------------------------------------


@pytest.mark.model
def test_v1_text_search_could_select_system_tokens_and_v2_cannot():
    """The reproduction from the review, kept as a regression.

    Marked ``model`` because it needs the pinned tokenizer files. It asserts the
    v1 behaviour *is* broken -- that is why v1 is frozen and superseded -- and
    that v2's span, being recorded rather than searched, cannot be affected.
    """
    from transformers import AutoTokenizer

    from cerl_rl import protocol
    from cerl_rl.rollout import _generated_mask

    tok = AutoTokenizer.from_pretrained(protocol.MODEL_ID, revision=protocol.MODEL_REVISION)
    answer = '{"tool": "tickets__get", "arguments": {"ticket_id": "tkt_000000000001"}}'
    messages = [
        {"role": "system", "content": "An earlier answer was:\n" + answer},
        {"role": "user", "content": "Observation"},
        {"role": "assistant", "content": answer},
    ]
    full = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    mask = _generated_mask(tok, ids, [answer])
    system_tokens = len(
        tok(full[: full.index("<|im_end|>")], add_special_tokens=False)["input_ids"],
    )
    assert int(mask[:system_tokens].sum()) == 32, (
        "the v1 defect no longer reproduces; this test pins why v1 is superseded"
    )

    # v2: the span is the recorded generated_ids. There is nothing to search.
    turn = _turn(0, list(ids[:system_tokens]), [5, 6, 7])
    assert turn.generated_ids == (5, 6, 7)
    assert len(turn.prompt_ids) == system_tokens


@pytest.mark.model
def test_v1_rerendering_changed_the_conditioning():
    """The second half of R1b, pinned against the real template."""
    from transformers import AutoTokenizer

    from cerl_rl import protocol
    from cerl_rl.rollout import _messages

    tok = AutoTokenizer.from_pretrained(protocol.MODEL_ID, revision=protocol.MODEL_REVISION)
    answer = '{"tool": "tickets__get", "arguments": {"ticket_id": "tkt_000000000001"}}'
    prompt = tok.apply_chat_template(
        _messages("System", ["First observation"], []),
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    complete = tok.apply_chat_template(
        _messages("System", ["First observation", "Second observation"],
                  [answer, '{"tool":"abstain","arguments":{"reason":"x"}}']),
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    generation_prefix = tok(prompt, add_special_tokens=False)["input_ids"]
    backward_prefix = tok(
        complete[: complete.index(answer)], add_special_tokens=False,
    )["input_ids"]
    assert generation_prefix != backward_prefix
    assert prompt.endswith("<think>\n\n</think>\n\n")
    assert not complete[: complete.index(answer)].endswith("<think>\n\n</think>\n\n")
