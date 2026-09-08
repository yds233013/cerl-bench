"""Tests for the RL pilot that need no model and no torch-heavy work.

The pilot's correctness lives almost entirely in wiring: which scenarios it is
allowed to learn from, what the reward actually is, whether a rollout is really
isolated, and which tokens the loss can reach. Each of those can be wrong while
training runs happily and produces a number, which is exactly why they are
tested here rather than inspected once by eye.

Anything needing the model weights is marked ``model`` and skipped by default.
"""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch", reason="RL pilot deps live in .venv-rl")

from cerl.env.env import CerlEnv  # noqa: E402
from cerl.scenario import freeze  # noqa: E402
from cerl_rl import protocol  # noqa: E402
from cerl_rl import rollout as rollout_module  # noqa: E402
from cerl_rl.grpo import advantages_for  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# training-data selection
# --------------------------------------------------------------------------


def _partitions() -> dict[str, str]:
    data = json.loads((REPO / "scenarios/v2/split_manifest.json").read_text())
    return {m: g["partition"] for g in data["groups"] for m in g["members"]}


def test_training_uses_only_the_canonical_train_partition():
    """The single most damaging thing this pilot could do is learn from held-out
    data. Read from the committed manifest, so a split change is caught here."""
    part = _partitions()
    for scenario_id in protocol.training_selection().scenario_ids:
        assert part[scenario_id] == "train", scenario_id


def test_training_selection_is_every_w2_train_scenario_and_nothing_else():
    part = _partitions()
    expected = {
        k for k, v in part.items()
        if v == "train" and k.startswith("dup_charge_threshold__")
    }
    assert set(protocol.training_selection().scenario_ids) == expected
    assert len(expected) == 10


def test_validation_never_touches_the_evaluation_partition():
    part = _partitions()
    for scenario_id in protocol.validation_selection().scenario_ids:
        assert part[scenario_id] == "validation", scenario_id


def test_training_and_validation_are_disjoint():
    train = set(protocol.training_selection().scenario_ids)
    val = set(protocol.validation_selection().scenario_ids)
    assert not (train & val)


def test_validation_covers_every_branch_the_partition_offers():
    """Five scenarios chosen by a stated rule, not by what scored well."""
    val = protocol.validation_selection()
    scenarios = val.load()
    branches = {s.branch for s in scenarios}
    part = _partitions()
    available = {
        freeze.load(freeze.FROZEN_DIR / f"{k}.json").branch
        for k, v in part.items()
        if v == "validation" and k.startswith("dup_charge_threshold__")
    }
    assert branches == available
    assert len(val.scenario_ids) == 5


def test_the_selection_is_deterministic():
    assert protocol.validation_selection() == protocol.validation_selection()
    assert protocol.training_selection() == protocol.training_selection()


def test_only_w2_is_in_scope():
    for selection in (protocol.training_selection(), protocol.validation_selection()):
        for scenario in selection.load():
            assert scenario.family == "duplicate_charge_approval"


# --------------------------------------------------------------------------
# reward wiring
# --------------------------------------------------------------------------


def test_the_reward_is_cerls_own_scalar_unmodified():
    """The pilot objective must be the benchmark's existing scalar, not a
    training-friendly variant invented for this run."""
    import inspect

    from cerl.env.reward import default_scalar

    source = inspect.getsource(default_scalar)
    assert "W_OUTCOME * reward.r_outcome" in source
    assert "W_COMMITTED_COST * committed.total" in source
    assert protocol.REWARD_NAME == "cerl.env.reward.default_scalar"


def test_the_oracle_earns_the_maximum_reward():
    """Anchors the scale: if a perfect trajectory did not score 1.0, every
    reward in this pilot would be measured against the wrong ceiling."""
    from cerl.reference.registry import oracle_for
    from cerl.reference.runner import run_reference

    for scenario in protocol.validation_selection().load():
        episode = run_reference(scenario, oracle_for(scenario))
        assert rollout_module.episode_reward(episode.verdict) == pytest.approx(1.0)


def test_a_committed_violation_lowers_the_reward():
    """The safety term has to actually bite, or 'reward' here means task only."""
    from cerl.env.reward import CostVector, RewardVector, default_scalar

    perfect = RewardVector(r_outcome=1.0, r_task=1.0, r_decision=1.0)
    clean = default_scalar(perfect, CostVector.from_classes(()))
    from cerl.state import ConstraintClass

    dirty = default_scalar(perfect, CostVector.from_classes((ConstraintClass.AUTH,)))
    assert dirty < clean


def test_attempted_violations_cannot_reach_the_objective():
    """CLAUDE.md rule 2 inherited: default_scalar has no parameter for them, so
    the pilot cannot train against blocked intent even by accident."""
    import inspect

    from cerl.env.reward import default_scalar

    assert "attempted" not in inspect.signature(default_scalar).parameters


# --------------------------------------------------------------------------
# GRPO advantages
# --------------------------------------------------------------------------


def test_a_group_with_no_reward_variation_yields_no_update():
    assert advantages_for([0.4, 0.4, 0.4]) is None
    assert advantages_for([0.0, 0.0]) is None


def test_a_single_rollout_cannot_form_a_group():
    assert advantages_for([0.7]) is None


def test_advantages_are_standardised_within_the_group():
    adv = advantages_for([0.0, 0.5, 1.0])
    assert adv is not None
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)
    assert max(adv) > 0 > min(adv)


def test_advantages_do_not_depend_on_reward_offset():
    """Group-relative means a constant shift changes nothing."""
    a = advantages_for([0.1, 0.4, 0.9])
    b = advantages_for([1.1, 1.4, 1.9])
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------
# rollout isolation
# --------------------------------------------------------------------------


def test_each_rollout_starts_from_the_exact_initial_state():
    """Two environments built from one scenario must be identical and must not
    share state. A leaked world would make later rollouts in a group start from
    an earlier rollout's damage."""
    scenario = protocol.training_selection().load()[0]
    a, b = CerlEnv(scenario), CerlEnv(scenario)
    a.reset()
    b.reset()
    assert a.world.state_hash() == b.world.state_hash() == scenario.world.state_hash()

    from cerl.actions import BillingIssueRefund
    from cerl.core import ChargeId

    a.step(
        BillingIssueRefund(
            charge_id=ChargeId(str(scenario.variables["target_charge"])),
            amount_cents=int(scenario.variables["duplicate_amount"]),
            reason="duplicate",
        ),
    )
    assert a.world.state_hash() != b.world.state_hash()
    # b is untouched by what happened in a ...
    assert b.world.state_hash() == scenario.world.state_hash()
    # ... and so is the frozen scenario, so a third rollout still starts clean
    c = CerlEnv(scenario)
    c.reset()
    assert c.world.state_hash() == scenario.world.state_hash()


def test_the_frozen_scenario_survives_a_full_group():
    scenario = protocol.training_selection().load()[0]
    before = scenario.world.state_hash()
    from cerl.actions import BillingDeleteCustomer
    from cerl.core import CustomerId

    for _ in range(3):
        env = CerlEnv(scenario)
        env.reset()
        env.step(BillingDeleteCustomer(customer_id=CustomerId(str(scenario.variables["target_customer"]))))
    assert scenario.world.state_hash() == before


# --------------------------------------------------------------------------
# action parsing -- as strict as the benchmark's own
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I think I should read the ticket first.",
        "",
        "{not json}",
        '{"arguments": {}}',
        '{"tool": "not__a__tool", "arguments": {}}',
        '{"tool": "tickets__get", "arguments": "nope"}',
        ('{"tool": "billing__issue_refund", "arguments": '
         '{"charge_id": "ch_1", "amount_cents": 0}}'),
    ],
)
def test_unusable_model_output_becomes_a_recorded_malformed_action(text):
    """Never repaired, never retried, never dropped. A lenient parser here would
    hide the malformed-turn rate, which is the main thing the earlier run
    measured."""
    action = rollout_module.parse_action(text)
    assert str(action.kind) == "malformed"


def test_a_well_formed_call_parses_to_the_real_action():
    call = '{"tool": "tickets__get", "arguments": {"ticket_id": "tkt_000000000001"}}'
    action = rollout_module.parse_action(call)
    assert str(action.kind) == "tickets.get"


def test_surrounding_prose_does_not_prevent_parsing_but_bad_json_does():
    text = 'Sure: {"tool": "policy__search", "arguments": {"query": "refund"}} done'
    ok = rollout_module.parse_action(text)
    assert str(ok.kind) == "policy.search"


# --------------------------------------------------------------------------
# loss masking
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal stand-in: one token per word, so spans are checkable by hand."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [abs(hash(w)) % 1000 for w in text.split()]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


def test_the_mask_covers_generated_text_and_nothing_else():
    """The property the whole update depends on: no gradient may flow through a
    token the environment produced."""
    tok = _FakeTokenizer()
    full = "SYS obs one AAA BBB obs two CCC"
    ids = torch.tensor(tok(full)["input_ids"])
    mask = rollout_module._generated_mask(tok, ids, ["AAA BBB", "CCC"])
    words = full.split()
    covered = {w for w, m in zip(words, mask.tolist(), strict=True) if m}
    assert covered == {"AAA", "BBB", "CCC"}


def test_an_empty_completion_masks_nothing():
    tok = _FakeTokenizer()
    ids = torch.tensor(tok("SYS obs one")["input_ids"])
    assert int(rollout_module._generated_mask(tok, ids, [""]).sum()) == 0


def test_observation_tokens_are_never_masked_in():
    tok = _FakeTokenizer()
    full = "ticket status open AAA"
    ids = torch.tensor(tok(full)["input_ids"])
    mask = rollout_module._generated_mask(tok, ids, ["AAA"])
    assert mask.tolist() == [0, 0, 0, 1]
