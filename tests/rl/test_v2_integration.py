"""Offline integration checks for the runnable v2 pilot.

Every test here drives ``run_pilot`` itself with a tiny CPU network and scripted
rollouts. No pretrained model is loaded, nothing is downloaded, and no training
run is started -- but the backward pass, the optimizer step, the record, the
replay verifier and the interruption paths are all real.

Each test corresponds to a defect that was reproduced before it was fixed; the
docstrings say what the old behaviour was, so a regression is recognisable
rather than merely red.
"""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="RL pilot deps live in .venv-rl")

from pydantic import TypeAdapter  # noqa: E402
from tests.rl.fakes import (  # noqa: E402
    ABSTAIN,
    ESCALATE,
    READ_POLICY,
    READ_TICKET,
    ScriptedTokenizer,
    TinyPolicy,
)

from cerl.actions import Action  # noqa: E402
from cerl.reference.runner import run_actions  # noqa: E402
from cerl.scenario import freeze  # noqa: E402
from cerl_rl import pilot_v2, protocol_v2  # noqa: E402
from cerl_rl.replay import verify_run  # noqa: E402
from cerl_rl.runtime import Deadline  # noqa: E402

_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(scope="module")
def train_scenario():
    return protocol_v2.training_selection().load()[:1]


@pytest.fixture(scope="module")
def val_scenario():
    return protocol_v2.validation_selection().load()[:1]


def _run(out, tokenizer, policy=None, *, updates=1, group_size=2, deadline=None,
         train=None, val=None, lr=1e-2):
    policy = policy or TinyPolicy()
    return policy, pilot_v2.run_pilot(
        policy, tokenizer, torch.device("cpu"), out=out,
        updates=updates, group_size=group_size, learning_rate=lr,
        deadline=deadline or Deadline(10_000),
        train_scenarios=train, val_scenarios=val, reserve_seconds=0.0,
    )


# --------------------------------------------------------------------------
# 1. a real backward pass and optimizer update
# --------------------------------------------------------------------------


def test_differing_rewards_produce_a_finite_nonzero_gradient_and_move_parameters(
    tmp_path, train_scenario, val_scenario,
):
    """Previously unreachable.

    Every fake rollout returned the same reply, so every group had equal rewards
    and was skipped -- ``run_pilot``'s backward and optimizer path had never
    executed in a test. Here the two rollouts in the group take *different*
    terminal actions, so the environment gives them different rewards.
    """
    tokenizer = ScriptedTokenizer([ABSTAIN, ABSTAIN, ESCALATE, *([ABSTAIN] * 40)])
    policy = TinyPolicy()
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}

    _policy, record = _run(tmp_path, tokenizer, policy,
                           train=train_scenario, val=val_scenario)

    group = record["groups"][0]
    assert not group["skipped"], group.get("skip_reason")
    assert len(set(group["rewards"])) > 1, "the fixture must produce differing rewards"
    assert group["advantages"] and abs(sum(group["advantages"])) < 1e-9

    grad_norm = group["grad_norm"]
    assert grad_norm is not None
    assert torch.isfinite(torch.tensor(grad_norm))
    assert grad_norm > 0.0, "a zero gradient is not a reward-driven update"
    assert group["generated_tokens"] > 0

    changed = [n for n, p in policy.named_parameters()
               if not torch.equal(before[n], p.detach())]
    assert len(changed) == len(before), f"only {len(changed)} of {len(before)} moved"
    assert record["reward_driven_updates"] == 1


def test_equal_rewards_skip_the_update_and_leave_parameters_untouched(
    tmp_path, train_scenario, val_scenario,
):
    """The other half: a group with no reward variation must change nothing."""
    tokenizer = ScriptedTokenizer([ABSTAIN] * 60)
    policy = TinyPolicy()
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}

    _policy, record = _run(tmp_path, tokenizer, policy,
                           train=train_scenario, val=val_scenario)

    group = record["groups"][0]
    assert group["skipped"]
    assert len(set(group["rewards"])) == 1
    assert group["grad_norm"] is None
    assert record["reward_driven_updates"] == 0
    for name, parameter in policy.named_parameters():
        assert torch.equal(before[name], parameter.detach()), name


def test_a_zero_gradient_is_not_counted_as_a_reward_driven_update(
    tmp_path, train_scenario, val_scenario, monkeypatch,
):
    """Rewards can differ while the gradient is still exactly zero -- for
    instance when the rollouts are token-identical and the advantages cancel.
    That step moves parameters only through optimizer state, so counting it
    would make "N reward-driven updates" mean less than it says."""
    import cerl_rl.pilot_v2 as module

    monkeypatch.setattr(module, "turn_backward", lambda *_a, **_k: (0.0, 4))
    tokenizer = ScriptedTokenizer([ABSTAIN, ABSTAIN, ESCALATE, *([ABSTAIN] * 40)])
    policy = TinyPolicy()
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}

    _policy, record = _run(tmp_path, tokenizer, policy,
                           train=train_scenario, val=val_scenario)

    group = record["groups"][0]
    assert group["skipped"]
    assert "zero" in group["skip_reason"]
    assert record["reward_driven_updates"] == 0
    for name, parameter in policy.named_parameters():
        assert torch.equal(before[name], parameter.detach())


# --------------------------------------------------------------------------
# 2. the replay verifier must understand the v2 format
# --------------------------------------------------------------------------


def _v2_run(tmp_path, train_scenario, val_scenario) -> pathlib.Path:
    tokenizer = ScriptedTokenizer(
        [READ_TICKET, ABSTAIN, READ_POLICY, ABSTAIN, ESCALATE, *([ABSTAIN] * 40)],
    )
    _run(tmp_path, tokenizer, train=train_scenario, val=val_scenario)
    return tmp_path / "pilot_run.json"


def test_a_real_v2_run_verifies(tmp_path, train_scenario, val_scenario):
    """It did not: ``verify_run`` required a ``protocol`` key that v2 records do
    not have, so every v2 run came back ``incomplete`` with zero episodes."""
    result = verify_run(_v2_run(tmp_path, train_scenario, val_scenario))
    assert result["format"] == "v2"
    assert result["status"] == "verified", result["mismatches"]
    assert result["ok"]
    assert result["episodes_replayed"] > 0


def test_the_historical_v1_run_still_verifies():
    """The corrected verifier must not invalidate the recorded experiment."""
    recorded = pathlib.Path("evidence/rl-pilot/pilot_run.json")
    if not recorded.exists():
        pytest.skip("no recorded pilot in this checkout")
    result = verify_run(recorded)
    assert result["format"] == "v1"
    assert result["status"] == "verified"
    assert result["episodes_replayed"] == 58


@pytest.mark.parametrize(
    ("label", "field", "value"),
    [
        ("reward", "reward", 999.0),
        ("trace hash", "trace_head_hash", "f" * 64),
        ("terminal hash", "terminal_state_hash", "f" * 64),
        ("task completion", "task_completion", 1.0),
        ("safety", "safe_completion", True),
        ("decision", "decision_correct", True),
        ("tool calls", "tool_calls", 99),
    ],
)
def test_an_altered_claim_fails_verification(
    tmp_path, train_scenario, val_scenario, label, field, value,
):
    path = _v2_run(tmp_path, train_scenario, val_scenario)
    record = json.loads(path.read_text())
    record["baseline"]["per_episode"][0][field] = value
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record))

    result = verify_run(tampered)
    assert not result["ok"], label
    assert field in {m["field"] for m in result["mismatches"]}, label


def test_an_action_appended_after_a_terminal_one_is_caught(
    tmp_path, train_scenario, val_scenario,
):
    """It was not: an action after a terminal one is never executed on replay,
    so every recomputed field still matched and the edit was invisible."""
    path = _v2_run(tmp_path, train_scenario, val_scenario)
    record = json.loads(path.read_text())
    record["baseline"]["per_episode"][0]["actions"].append(
        {"kind": "abstain", "reason": "appended"},
    )
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record))

    result = verify_run(tampered)
    assert not result["ok"]
    assert "actions" in {m["field"] for m in result["mismatches"]}


# --------------------------------------------------------------------------
# 3. durability: turns, checkpoints, and the deadline
# --------------------------------------------------------------------------


def test_turns_of_an_interrupted_episode_survive_and_replay(
    tmp_path, train_scenario, val_scenario,
):
    """Previously all of them were lost.

    Turns were written only after a whole episode returned, so an episode
    interrupted part way through persisted nothing -- discarding turns the
    environment really executed.
    """
    clock = FakeClock()
    policy = TinyPolicy()
    policy.on_generate = lambda n: clock.advance(1000) if n >= 3 else None
    tokenizer = ScriptedTokenizer(
        [READ_TICKET, READ_POLICY, READ_TICKET, READ_POLICY, ABSTAIN] * 20,
    )
    _policy, record = _run(
        tmp_path, tokenizer, policy, deadline=Deadline(500, clock=clock),
        train=train_scenario, val=val_scenario,
    )
    assert record["interruption"] is not None

    lines = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert len(lines) == 3, f"expected the 3 completed turns, got {len(lines)}"

    for turn in lines:
        assert turn["prompt_ids"] and turn["generated_ids"]
        assert turn["stop_reason"] in {"eos", "length", "stopped"}
        assert turn["outcome"], "the environment outcome must be recorded"
        assert turn["action"]["kind"] == turn["action_kind"]

    # and they replay: the recorded actions are complete, not just kinds
    scenario = freeze.load(freeze.FROZEN_DIR / f"{lines[0]['scenario_id']}.json")
    actions = tuple(_ADAPTER.validate_python(t["action"]) for t in lines)
    episode = run_actions(scenario, actions)
    assert len(episode.trace.agent_entries()) == len(actions)
    assert [str(e.outcome) for e in episode.trace.agent_entries()] == [
        t["outcome"] for t in lines
    ]


def test_a_checkpoint_from_a_completed_update_survives_a_later_interruption(
    tmp_path, train_scenario, val_scenario,
):
    """Checkpoints were written only after the whole training loop, so an
    interruption discarded every update that had already been applied."""
    clock = FakeClock()
    policy = TinyPolicy()
    calls = {"n": 0}

    def creep(_n: int) -> None:
        calls["n"] += 1
        if calls["n"] > 5:          # let update 0 finish, then overrun
            clock.advance(5000)

    policy.on_generate = creep
    tokenizer = ScriptedTokenizer(
        [ABSTAIN, ABSTAIN, ESCALATE] + [ABSTAIN, ESCALATE] * 40,
    )
    _policy, record = _run(
        tmp_path, tokenizer, policy, updates=3, deadline=Deadline(2000, clock=clock),
        train=train_scenario, val=val_scenario,
    )
    assert record["interruption"] is not None
    assert record["reward_driven_updates"] >= 1
    assert record["checkpoint"]["sha256"], "no checkpoint survived the interruption"
    # It names the last update that actually completed, whichever that was.
    completed = [g["update"] for g in record["groups"] if not g["skipped"]]
    assert record["checkpoint"]["after_update"] == completed[-1]
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert policy.saved_checkpoints, "save_pretrained was never called"


def test_the_deadline_is_checked_immediately_before_the_optimizer_step(tmp_path):
    """An overrun between the backward pass and the update would otherwise leave
    gradients applied with no record of the stop."""
    import inspect

    source = inspect.getsource(pilot_v2.run_pilot)
    step = source.index("optimizer.step()")
    preceding = source[:step]
    assert "optimizer step" in preceding, "no deadline check guards optimizer.step()"
    assert preceding.rindex("deadline.enforce") > preceding.rindex("turn_backward")


def test_the_time_limit_includes_model_loading(tmp_path):
    """The budget was constructed as an argument to run_pilot, so Python
    evaluated ``load_policy(...)`` first and loading was free."""
    import inspect

    source = inspect.getsource(pilot_v2.main)
    # Comments mention these names too, so compare statements only.
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert code.index("deadline = Deadline(") < code.index("policy = load_policy("), (
        "the deadline must start before the model is loaded"
    )
    assert 'deadline.enforce("model loading")' in code


def test_an_interruption_is_recorded_before_anything_else_is_attempted(
    tmp_path, train_scenario, val_scenario,
):
    clock = FakeClock()
    policy = TinyPolicy()
    policy.on_generate = lambda n: clock.advance(10_000)
    tokenizer = ScriptedTokenizer([ABSTAIN] * 40)
    _policy, record = _run(
        tmp_path, tokenizer, policy, deadline=Deadline(50, clock=clock),
        train=train_scenario, val=val_scenario,
    )
    saved = json.loads((tmp_path / "pilot_run.json").read_text())
    assert saved["interruption"]["phase"] == record["interruption"]["phase"]
    assert saved["groups"] == []
    assert "elapsed_seconds" in saved["interruption"]
