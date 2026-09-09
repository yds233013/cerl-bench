"""Model-dependent checks. Opt-in.

These load real weights and run real forward passes, so they are slow and need
the isolated ``.venv-rl``. They are marked ``model`` and deselected by default:

    PYTHONPATH=src .venv-rl/bin/python -m pytest tests/rl -m model
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="RL pilot deps live in .venv-rl")
pytest.importorskip("peft")

from cerl_rl import protocol  # noqa: E402
from cerl_rl import rollout as rollout_module  # noqa: E402
from cerl_rl.grpo import advantages_for, group_backward  # noqa: E402
from cerl_rl.model import (  # noqa: E402
    adapter_state,
    load_policy,
    load_tokenizer,
    pick_device,
)

pytestmark = pytest.mark.model

_SYSTEM = "You are a support agent." + (
    rollout_module.SYSTEM_SUFFIX % rollout_module.tool_menu()
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def policy():
    return load_policy(pick_device(), lora_rank=4)


def test_thinking_is_disabled_by_the_template_not_by_a_flag(tokenizer):
    """The exact mistake the earlier 4B run made, pinned so it cannot recur.

    A ``think`` flag on a *runner* changes nothing about generation. What
    actually suppresses the reasoning block is the chat template: with
    ``enable_thinking=False`` it appends an already-closed ``<think></think>``
    pair to the prompt, so generation begins after it. Asserted on the rendered
    string, because that is the only thing the model ever sees.
    """
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    on = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    off = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    default = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    assert "<think>" not in on
    assert default == on, "the default leaves thinking on; it is not a safe default here"
    assert off.endswith("<think>\n\n</think>\n\n"), off[-40:]


def test_the_policy_input_contains_no_privileged_information(tokenizer, policy):
    """Branch, required decision, rubric and permitted diffs must never reach
    the prompt. Checked on the rendered text, not on imports."""
    scenario = protocol.validation_selection().load()[0]
    from cerl.env.env import CerlEnv
    from cerl.env.render import render_observation

    env = CerlEnv(scenario)
    observation = env.reset()
    suffix = rollout_module.SYSTEM_SUFFIX % rollout_module.tool_menu()
    rendered = render_observation(observation) + suffix

    # The branch name and every rubric predicate id are the answer key.
    assert scenario.branch not in rendered
    for item in scenario.rubric:
        assert item.id not in rendered
    # Axis values are labels too: "appr-expired" would tell the policy which
    # counterfactual it is in.
    for axis_value in scenario.axes.values():
        assert f"appr-{axis_value}" not in rendered

    # Note on what is deliberately NOT asserted: the word "escalate" does appear,
    # because escalate is an available action and its tool has to be listed. A
    # test forbidding it would be checking the wrong thing -- the leak that
    # matters is being told which decision is *correct*, not which are possible.


def test_a_rollout_masks_only_generated_tokens(tokenizer, policy):
    """End to end on the real model: the environment writes most of the
    sequence, and none of it may be trainable."""
    scenario = protocol.validation_selection().load()[0]
    tokenised = rollout_module.rollout(
        policy, tokenizer, scenario, max_actions=2, max_new_tokens=64,
        temperature=0.0, device=pick_device(),
        system_prompt=_SYSTEM,
        max_prompt_tokens=protocol.MAX_PROMPT_TOKENS,
    )
    total = int(tokenised.input_ids.numel())
    generated = tokenised.n_generated_tokens
    assert 0 < generated < total
    # The observations dominate; if the mask ever covered most of the sequence
    # it would mean the model was being trained on its own inputs.
    assert generated / total < 0.5


def test_a_reward_driven_step_changes_the_adapter_and_stays_finite(tokenizer, policy):
    """One real GRPO update, with hand-made advantages so the test does not
    depend on the model happening to produce reward variation."""
    scenario = protocol.validation_selection().load()[0]
    device = pick_device()
    episodes = [
        rollout_module.rollout(policy, tokenizer, scenario, max_actions=2, max_new_tokens=48,
                  temperature=1.0, device=device,
                  system_prompt=_SYSTEM,
                  max_prompt_tokens=protocol.MAX_PROMPT_TOKENS)
        for _ in range(2)
    ]
    before = adapter_state(policy)
    policy.train()
    _loss, tokens = group_backward(policy, episodes, [1.0, -1.0], device)
    assert tokens > 0
    grads = [p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradient reached the adapter"
    norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    assert torch.isfinite(norm) and norm > 0

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0,
    )
    optimizer.step()
    after = adapter_state(policy)
    delta = sum(float((after[k] - before[k]).abs().sum()) for k in before)
    assert delta > 0, "the optimizer step did not move the adapter"


def test_a_zero_variance_group_is_refused_before_any_step():
    """The guard that keeps 'ran 20 iterations' from meaning 'learned nothing'."""
    assert advantages_for([0.25, 0.25, 0.25, 0.25]) is None
