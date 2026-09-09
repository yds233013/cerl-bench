"""The recorded experiment must still say what the report says it says.

These read committed evidence only -- no model, no torch, no inference -- so
they keep working long after the run, and they fail loudly if a number in the
report drifts away from the record it came from.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest
from pydantic import TypeAdapter

from cerl.actions import Action
from cerl.env.reward import W_COMMITTED_COST, W_DECISION, W_OUTCOME, W_TASK
from cerl.reference.runner import run_actions
from cerl.scenario import freeze

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "evidence" / "rl-pilot"
SMOKE = pathlib.Path(__file__).resolve().parents[2] / "evidence" / "rl-v2-smoke-2rollout"
_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

#: The v1 pilot's checkpoint is deliberately **not committed** -- it is a model
#: weight file, excluded by ``.gitignore`` along with the base model and the
#: caches. So the checks that read it can only run where the original run
#: happened. They skip rather than fail on a fresh clone, and the committed v2
#: smoke adapter carries the equivalent check that *does* run everywhere.
_V1_WEIGHTS = EVIDENCE / "adapter" / "adapter_model.safetensors"
_needs_v1_checkpoint = pytest.mark.skipif(
    not _V1_WEIGHTS.exists(),
    reason=(
        "the v1 pilot checkpoint is not committed (model weights are gitignored); "
        "this check runs only on a machine holding the original run"
    ),
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads((EVIDENCE / "pilot_run.json").read_text())


def test_the_recorded_update_count_matches_the_recorded_groups(record):
    driven = [g for g in record["groups"] if not g["skipped"]]
    assert record["reward_driven_updates"] == len(driven) == 8
    assert len(record["groups"]) == 12
    assert sum(1 for g in record["groups"] if g["skipped"]) == 4


def test_every_skipped_group_really_had_no_reward_variation(record):
    """The claim that skipping was principled, checked against the rewards."""
    for group in record["groups"]:
        if group["skipped"]:
            assert len(set(group["rewards"])) == 1, group["update"]
            assert group["grad_norm"] is None


def test_every_reward_driven_group_had_a_finite_gradient(record):
    for group in record["groups"]:
        if not group["skipped"]:
            assert group["grad_norm"] is not None
            assert 0.0 < group["grad_norm"] < 1e3
            assert group["generated_tokens"] > 0


@_needs_v1_checkpoint
def test_the_checkpoint_on_disk_is_the_one_the_record_names(record):
    digest = hashlib.sha256(_V1_WEIGHTS.read_bytes()).hexdigest()
    assert digest == record["checkpoint"]["sha256"]


def test_the_committed_smoke_checkpoint_is_the_one_its_record_names():
    """The v2 smoke adapter *is* committed, so this runs on any clone. It is the
    published claim that the checkpoint in the tree is the one the run wrote."""
    record = json.loads((SMOKE / "smoke.json").read_text())
    weights = SMOKE / "adapter" / record["checkpoint"]["weights_file"]
    assert weights.exists()
    assert weights.stat().st_size == record["checkpoint"]["bytes"]
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == record["checkpoint"]["sha256"]


@_needs_v1_checkpoint
def test_the_saved_adapter_is_trained_rather_than_freshly_initialised():
    """A fresh LoRA has ``lora_B == 0`` exactly. Non-zero here is direct evidence
    that something was learned -- and that the baseline measured the base model,
    because a zero ``lora_B`` contributes nothing."""
    safetensors = pytest.importorskip("safetensors.torch")
    state = safetensors.load_file(str(EVIDENCE / "adapter" / "adapter_model.safetensors"))
    b_tensors = [v for k, v in state.items() if "lora_B" in k]
    assert len(b_tensors) == 112
    assert all(float(v.abs().sum()) > 0 for v in b_tensors)


def test_every_recorded_reward_is_reproduced_by_replaying_its_actions(record):
    """The strongest statement available offline: each reward is re-derivable
    from the recorded actions through the ordinary verifier."""
    checked = 0
    for group in record["groups"]:
        scenario = freeze.load(freeze.FROZEN_DIR / f"{group['scenario_id']}.json")
        for episode in group["episodes"]:
            actions = tuple(_ADAPTER.validate_python(a) for a in episode["actions"])
            verdict = run_actions(scenario, actions).verdict
            expected = (
                W_OUTCOME * (1.0 if verdict.correct_final_state else 0.0)
                + W_TASK * float(verdict.task_completion)
                + W_DECISION * (1.0 if verdict.decision_correct else 0.0)
                - W_COMMITTED_COST * len(verdict.violations)
            )
            assert expected == pytest.approx(episode["reward"], abs=1e-9)
            checked += 1
    assert checked == 48


def test_the_reward_variation_came_from_the_decision_term(record):
    """What the pilot actually trained on.

    Every rollout had ``correct_final_state`` false and zero committed
    violations, so the outcome and cost terms were constant at 0. The spread is
    the 0.3 decision term plus a small task term -- the signal was 'declare the
    right kind of outcome', not 'complete the work'.
    """
    audit = json.loads((EVIDENCE / "rollout_audit.json").read_text())
    assert len(audit) == 48
    assert all(row["outcome"] == 0.0 for row in audit)
    assert all(row["cost"] == 0.0 for row in audit)
    correct = [r for r in audit if r["decision_correct"]]
    wrong = [r for r in audit if not r["decision_correct"]]
    assert len(correct) == 11
    assert min(r["reward"] for r in correct) >= 0.3
    assert max(r["reward"] for r in wrong) < 0.3


def test_no_training_rollout_committed_or_attempted_a_violation(record):
    audit = json.loads((EVIDENCE / "rollout_audit.json").read_text())
    assert sum(len(r["committed"]) for r in audit) == 0
    assert sum(len(r["attempted"]) for r in audit) == 0


def test_before_and_after_produced_identical_action_sequences(record):
    """The headline negative result, pinned."""
    before, after = record["baseline"]["per_episode"], record["after"]["per_episode"]
    assert len(before) == len(after) == 5
    for b, a in zip(before, after, strict=True):
        assert b["scenario_id"] == a["scenario_id"]
        assert b["actions"] == a["actions"]
        assert b["trace_head_hash"] == a["trace_head_hash"]


def test_the_feasibility_record_matches_the_protocol(record):
    feas = json.loads((EVIDENCE / "feasibility.json").read_text())
    assert feas["summary"]["train"]["safe_at_limit_12"] == 7
    assert feas["summary"]["validation"]["safe_at_limit_12"] == 2
    assert feas["summary"]["train"]["safe_full_gold"] == 10
    assert feas["summary"]["validation"]["safe_full_gold"] == 5


# --------------------------------------------------------------------------
# R6: tool-call metrics
# --------------------------------------------------------------------------


def test_the_runner_counted_terminal_declarations_as_tool_calls(record):
    """The defect, pinned. The recorded rows keep their values; this asserts the
    discrepancy exists so the derived audit is not mistaken for a restatement."""
    from cerl_rl.audit import audit

    result = audit(record)
    baseline = result["phases"]["baseline"]
    assert baseline["reported_tool_calls"] == 15
    assert baseline["verifier_tool_calls"] == 10
    assert baseline["terminal_declarations"] == 5

    training = result["phases"]["training"]
    assert training["reported_tool_calls"] == 191
    assert training["verifier_tool_calls"] == 150
    assert training["terminal_declarations"] == 41
    assert training["malformed_actions"] == 26
    assert training["total_actions"] == 217


def test_the_corrected_categories_agree_with_the_verifier(record):
    """Two independent routes to the same number: categorising the recorded
    actions, and asking the verifier. They must agree, or the categorisation is
    the wrong definition rather than a corrected one."""
    from cerl_rl.audit import audit

    for phase in audit(record)["phases"].values():
        assert phase["tool_calls"] == phase["verifier_tool_calls"]
        assert (
            phase["tool_calls"] + phase["terminal_declarations"]
            + phase["malformed_actions"] == phase["total_actions"]
        )


def test_the_original_record_is_not_modified_by_auditing(record):
    from cerl_rl.audit import audit

    before = json.dumps(record, sort_keys=True)
    audit(record)
    assert json.dumps(record, sort_keys=True) == before
