"""The runnable v2 pilot: it must use the corrected path, and only that path.

Driven end to end with a fake model and tokenizer. No weights are loaded, no
inference runs, and the whole loop -- rollouts, advantages, backward, records,
deadline behaviour -- is exercised in under a second.
"""

from __future__ import annotations

import ast
import json
import pathlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="RL pilot deps live in .venv-rl")
from torch import nn  # noqa: E402

from cerl_rl import pilot_v2, protocol_v2  # noqa: E402
from cerl_rl.runtime import Deadline  # noqa: E402

SOURCE = pathlib.Path(pilot_v2.__file__)


# --------------------------------------------------------------------------
# it must not reach for the v1 path -- checked structurally, not by substring
# --------------------------------------------------------------------------


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.add(f"{module}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


@pytest.fixture(scope="module")
def tree() -> ast.AST:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_the_v2_pilot_never_calls_the_v1_rollout_or_group_backward(tree):
    called = _called_names(tree)
    assert "rollout" not in called, "v1 rollout is called"
    assert "group_backward" not in called, "v1 group_backward is called"
    assert "rollout_v2" in called
    assert "turn_backward" in called


def test_the_v2_pilot_does_not_import_the_v1_rollout_or_group_backward(tree):
    imported = _imported_names(tree)
    assert "cerl_rl.rollout.rollout" not in imported
    assert "cerl_rl.rollout" not in imported
    assert "cerl_rl.grpo.group_backward" not in imported
    assert "cerl_rl.rollout_v2.rollout_v2" in imported
    assert "cerl_rl.grpo.turn_backward" in imported


def test_the_v2_pilot_reads_limits_from_protocol_v2_only(tree):
    imported = _imported_names(tree)
    assert "cerl_rl.protocol_v2" in imported or any(
        name.startswith("cerl_rl.protocol_v2") for name in imported
    )
    assert "cerl_rl.protocol" not in imported, "v1 protocol constants leaked in"


def test_v1_remains_untouched_and_still_uses_its_own_path():
    """The historical runner must keep working exactly as it did."""
    v1 = ast.parse(pathlib.Path(pilot_v2.__file__).with_name("pilot.py").read_text())
    called = _called_names(v1)
    assert "group_backward" in called
    assert "turn_backward" not in called


# --------------------------------------------------------------------------
# a fake policy: no weights, no inference
# --------------------------------------------------------------------------


class FakeTokenizer:
    """Enough of a tokenizer to drive the loop deterministically."""

    eos_token_id = 2
    pad_token_id = 2

    def __init__(self) -> None:
        self.additional_special_tokens_ids: list[int] = []
        self._script: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, **_kwargs):
        return "|".join(f"{m['role']}:{m['content'][:40]}" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [(abs(hash(word)) % 50) + 3 for word in text.split("|")]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids}

    _FALLBACK = '{"tool":"abstain","arguments":{"reason":"x"}}'

    def decode(self, ids, skip_special_tokens=True):
        return self._script.pop(0) if self._script else self._FALLBACK


class FakePolicy(nn.Module):
    """A tiny network with the shapes ``turn_backward`` expects."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            __call__=None,
        )
        self._backbone = nn.Embedding(64, 8)
        self._mix = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 64, bias=False)
        self.replies = list(replies)
        self.generate_calls = 0
        # ``turn_backward`` resolves ``causal_lm(model).model`` / ``.lm_head``
        self.model = SimpleNamespace(
            __call__=lambda **kw: SimpleNamespace(
                last_hidden_state=torch.tanh(self._mix(self._backbone(kw["input_ids"]))),
            ),
        )

    def __call__(self, **kwargs):  # pragma: no cover - not used by turn_backward
        raise AssertionError("the loss must go through .model and .lm_head")

    def generate(self, *, input_ids, max_new_tokens, **_kwargs):
        self.generate_calls += 1
        return torch.cat(
            [input_ids, torch.tensor([[7, 9, 2]])], dim=1,
        )

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def save_pretrained(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(b"fake-adapter")


def _tokenizer(replies: list[str]) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    tokenizer._script = list(replies)
    return tokenizer


@pytest.fixture
def scenarios():
    return protocol_v2.validation_selection().load()[:2]


def _run(tmp_path, replies, *, updates=1, group_size=2, seconds=1000.0, clock=None, scenarios=None):
    policy = FakePolicy(replies)
    tokenizer = _tokenizer(replies * 50)
    deadline = Deadline(seconds, clock=clock) if clock else Deadline(seconds)
    return policy, pilot_v2.run_pilot(
        policy, tokenizer, torch.device("cpu"),
        out=tmp_path, updates=updates, group_size=group_size, learning_rate=1e-4,
        deadline=deadline, train_scenarios=scenarios, val_scenarios=scenarios,
        reserve_seconds=0.0,
    )


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_the_v2_pilot_runs_end_to_end_without_any_model_weights(tmp_path, scenarios):
    replies = ['{"tool":"policy__search","arguments":{"query":"refund"}}']
    policy, record = _run(tmp_path, replies, scenarios=scenarios)

    assert policy.generate_calls > 0
    assert record["protocol_version"] == protocol_v2.PROTOCOL_VERSION
    assert record["uses"]["rollout"] == "cerl_rl.rollout_v2.rollout_v2"
    assert record["uses"]["loss"] == "cerl_rl.grpo.turn_backward"
    assert "baseline" in record and "after" in record
    assert (tmp_path / "pilot_run.json").exists()


def test_the_run_record_carries_the_v2_limits(tmp_path, scenarios):
    _policy, record = _run(tmp_path, ['{"tool":"abstain","arguments":{"reason":"x"}}'],
                           scenarios=scenarios)
    assert record["limits"] == {
        "max_actions": 24, "max_prompt_tokens": 8192, "max_new_tokens": 160,
    }
    assert record["sampling"]["top_k"] == 0
    assert record["sampling"]["top_p"] == 1.0


def test_every_turn_persists_its_exact_tokens(tmp_path, scenarios):
    _policy, _record = _run(tmp_path, ['{"tool":"abstain","arguments":{"reason":"x"}}'],
                            scenarios=scenarios)
    lines = (tmp_path / "turns.jsonl").read_text().splitlines()
    assert lines, "no per-turn provenance was written"
    for line in lines:
        turn = json.loads(line)
        for field in ("prompt_ids", "generated_ids", "stop_reason",
                      "action_kind", "outcome", "scenario_id", "step_index"):
            assert field in turn, field
        assert turn["prompt_ids"], "prompt tokens must be recorded"
        assert turn["generated_ids"] == [7, 9, 2]
        assert turn["stop_reason"] in {"eos", "length", "stopped"}
        assert turn["outcome"], "the environment outcome must be recorded"


def test_the_generated_span_is_recorded_not_reconstructed(tmp_path, scenarios):
    """The whole point of v2: the ids written are the ids sampled."""
    _policy, _record = _run(tmp_path, ['{"tool":"abstain","arguments":{"reason":"x"}}'],
                            scenarios=scenarios)
    turn = json.loads((tmp_path / "turns.jsonl").read_text().splitlines()[0])
    # the fake generator always appends exactly these three tokens
    assert turn["generated_ids"] == [7, 9, 2]
    # and the prompt is whatever the tokenizer produced, not a re-render
    assert len(turn["prompt_ids"]) > 0


def test_metrics_separate_tool_calls_from_terminal_declarations(tmp_path, scenarios):
    """R6 carried into the v2 record."""
    _policy, record = _run(tmp_path, ['{"tool":"abstain","arguments":{"reason":"x"}}'],
                           scenarios=scenarios)
    baseline = record["baseline"]
    assert "terminal_declarations" in baseline
    assert "tool_calls" in baseline
    for row in baseline["per_episode"]:
        assert row["tool_calls"] + row["terminal_declarations"] + row["malformed_actions"] \
            == row["total_actions"]


def test_termination_distinguishes_declaring_from_running_out(tmp_path, scenarios):
    """R9: ``step_limited`` collapsed two different events."""
    _policy, record = _run(tmp_path, ['{"tool":"abstain","arguments":{"reason":"x"}}'],
                           scenarios=scenarios)
    for row in record["baseline"]["per_episode"]:
        assert row["termination"] in {"declared", "action_limit", "context_exhausted"}
    assert record["baseline"]["terminations"]


# --------------------------------------------------------------------------
# the deadline bounds every operation
# --------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_an_overrun_during_baseline_is_recorded_before_anything_else_starts(
    tmp_path, scenarios,
):
    clock = FakeClock()
    policy = FakePolicy([])
    tokenizer = _tokenizer(['{"tool":"abstain","arguments":{"reason":"x"}}'] * 50)

    original = policy.generate

    def slow_generate(**kwargs):
        clock.advance(100)
        return original(**kwargs)

    policy.generate = slow_generate
    record = pilot_v2.run_pilot(
        policy, tokenizer, torch.device("cpu"),
        out=tmp_path, updates=2, group_size=2, learning_rate=1e-4,
        deadline=Deadline(60, clock=clock),
        train_scenarios=scenarios, val_scenarios=scenarios, reserve_seconds=0.0,
    )
    assert record["interruption"] is not None
    assert "baseline" in record["interruption"]["phase"]
    assert record["stopped_because"].startswith("interrupted")
    # and it was written to disk before returning
    saved = json.loads((tmp_path / "pilot_run.json").read_text())
    assert saved["interruption"]["phase"] == record["interruption"]["phase"]
    # training never began
    assert saved["groups"] == []


def test_a_generation_overrun_mid_training_stops_and_preserves_prior_work(
    tmp_path, scenarios,
):
    clock = FakeClock()
    policy = FakePolicy([])
    # A terminal reply, so each episode is exactly one generation: the two
    # baseline scenarios use calls 1-2, and training begins at call 3.
    tokenizer = _tokenizer(['{"tool":"abstain","arguments":{"reason":"x"}}'] * 200)
    original = policy.generate
    state = {"calls": 0}

    def creeping_generate(**kwargs):
        state["calls"] += 1
        if state["calls"] > 2:      # let the baseline finish, then overrun
            clock.advance(500)
        return original(**kwargs)

    policy.generate = creeping_generate
    record = pilot_v2.run_pilot(
        policy, tokenizer, torch.device("cpu"),
        out=tmp_path, updates=3, group_size=2, learning_rate=1e-4,
        deadline=Deadline(400, clock=clock),
        train_scenarios=scenarios, val_scenarios=scenarios, reserve_seconds=0.0,
    )
    assert record["interruption"] is not None
    assert "generation" in record["interruption"]["phase"]
    # the baseline that did complete is preserved
    assert record["baseline"]["episodes"] == len(scenarios)
    # and so is the per-turn provenance collected before the stop
    assert (tmp_path / "turns.jsonl").read_text().strip()


def test_the_deadline_is_checked_before_each_generation_not_only_per_episode(
    tmp_path, scenarios,
):
    """A single generation is the longest operation in the run; a budget
    consulted only between episodes cannot bound it."""
    import inspect

    training = inspect.getsource(pilot_v2.run_pilot)
    evaluation = inspect.getsource(pilot_v2.evaluate_v2)
    # training rollouts
    assert "before_generate=" in training
    assert "deadline.enforce" in training
    # evaluation rollouts, per scenario and per generation
    assert "before_generate=" in evaluation
    assert evaluation.count("deadline.enforce") >= 2
    # and the backward pass
    assert "backward" in training
