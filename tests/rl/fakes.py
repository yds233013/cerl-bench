"""Synthetic policies for driving the v2 pilot without a pretrained model.

Deliberately real ``nn.Module``s rather than stand-ins: the loss backwards
through ``inner.model(...)`` and ``inner.lm_head``, so anything that merely
looks like a model from the outside will pass a smoke test and fail the moment
a gradient is actually taken. An earlier fixture used ``SimpleNamespace`` for
``.model`` and looked fine, because no test had ever reached the backward path.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import torch
from torch import nn

VOCAB = 64
HIDDEN = 8


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.mix = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(self.embed(input_ids))))


class TinyPolicy(nn.Module):
    """A real network with the surface ``turn_backward`` and the rollout need.

    ``generate`` appends a fixed span so the recorded generated ids are known
    exactly; the *text* of each turn comes from the tokenizer's script, which is
    what steers the environment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.generate_calls = 0
        self.saved_checkpoints: list[pathlib.Path] = []
        self.on_generate = None
        self.eos_id = 2

    def generate(self, *, input_ids: torch.Tensor, max_new_tokens: int, **_kwargs):
        """Append a span that differs per call.

        A fixed span would make two rollouts in a group token-identical, so
        their losses would cancel exactly and the gradient would be zero for
        arithmetic reasons that have nothing to do with the pilot. Real
        generation produces different tokens for different actions; this
        mirrors that minimally.
        """
        self.generate_calls += 1
        if self.on_generate is not None:
            self.on_generate(self.generate_calls)
        base = 10 + (self.generate_calls * 3) % (VOCAB - 20)
        span = [base, base + 1, self.eos_id]
        return torch.cat([input_ids, torch.tensor([span])], dim=1)

    def save_pretrained(self, path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(
            b"fake-adapter-" + str(self.generate_calls).encode(),
        )
        self.saved_checkpoints.append(path)


class ScriptedTokenizer:
    """Turns each generation into a chosen action text.

    The script is a flat queue: one entry per turn, in the order the pilot
    generates them. That is what lets a test give two rollouts in the same group
    genuinely different environment rewards.
    """

    eos_token_id = 2
    pad_token_id = 2
    _FALLBACK = '{"tool":"abstain","arguments":{"reason":"fallback"}}'

    def __init__(self, script: list[str] | None = None) -> None:
        self.additional_special_tokens_ids: list[int] = []
        self.script = list(script or [])
        self.decoded: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, **_kwargs) -> str:
        return "|".join(f"{m['role']}:{m['content'][:40]}" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [(abs(hash(word)) % (VOCAB - 3)) + 3 for word in text.split("|")]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        text = self.script.pop(0) if self.script else self._FALLBACK
        self.decoded.append(text)
        return text


# Convenience action texts used across the tests.
ABSTAIN = '{"tool":"abstain","arguments":{"reason":"nothing to do"}}'
ESCALATE = '{"tool":"escalate","arguments":{"reason":"needs a human","to":"billing-approvals"}}'
FINISH = '{"tool":"finish","arguments":{"summary":"done"}}'
READ_TICKET = '{"tool":"tickets__get","arguments":{"ticket_id":"tkt_000000000001"}}'
READ_POLICY = '{"tool":"policy__search","arguments":{"query":"refund threshold"}}'
