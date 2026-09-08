"""v2 rollouts: exact token provenance, exact conditioning.

The defect this exists to fix is subtle and was invisible in every summary
number. v1 trained on a span located by **searching** the re-rendered
conversation for the decoded completion text. Two things go wrong:

* the search is over the whole sequence from position zero, so it can select
  text in the system prompt or an observation that happens to match -- 32 tokens
  in the reproduced case; and
* the re-render is not the string the model was sampled under, because Qwen's
  chat template drops the empty ``<think></think>`` block from *earlier*
  assistant turns. The loss then evaluates a different conditional than
  generation did.

v2 never reconstructs. Each ``generate`` call's exact prompt ids and generated
ids are kept, and the loss re-runs precisely that prompt. A turn is scored under
the context it was actually sampled in, and the trained positions are known
rather than found.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cerl.actions import Action
from cerl.core import Frozen
from cerl.env.render import render_observation
from cerl.scenario.schema import FrozenScenario
from cerl.verify.verifier import verify
from cerl_rl import protocol_v2
from cerl_rl.environment import ActionCategory, categorise, drive, episode_reward, parse_action
from cerl_rl.rollout import Episode
from cerl_rl.tools import tool_contract


class TokenProvenanceError(RuntimeError):
    """A turn's recorded tokens are not self-consistent.

    Raised rather than worked around. Every alternative -- skipping the turn,
    truncating it, falling back to a search -- produces a training step whose
    trained span is unknown, which is the exact failure v2 exists to remove.
    """


class TurnRecord(Frozen, arbitrary_types_allowed=True):
    """One assistant turn, as tokens, exactly as it happened."""

    step_index: int
    #: The prompt the model was actually conditioned on, token for token.
    prompt_ids: tuple[int, ...]
    #: What it generated, before any decoding or special-token stripping.
    generated_ids: tuple[int, ...]
    #: ``eos`` -- the model chose to stop. ``length`` -- it hit ``max_new_tokens``
    #: and was cut off. These are different events and are recorded as such:
    #: a truncated turn is not a decision to stop.
    stop_reason: str
    text: str
    action_kind: str
    category: str
    outcome: str

    @property
    def sequence(self) -> tuple[int, ...]:
        return (*self.prompt_ids, *self.generated_ids)


class EpisodeV2(Frozen, arbitrary_types_allowed=True):
    turns: tuple[TurnRecord, ...]
    episode: Episode
    #: ``declared`` | ``action_limit`` | ``context_exhausted``.
    termination: str = "declared"

    @property
    def generated_tokens(self) -> int:
        return sum(len(t.generated_ids) for t in self.turns)


def system_prompt_v2(base: str) -> str:
    """Base instructions plus the **complete** public tool contract."""
    return (
        base
        + "\n\nReply with ONE JSON object and nothing else. No prose, no "
        "explanation, no markdown fence. The object has exactly two keys:\n\n"
        '  {"tool": "<tool name>", "arguments": {<arguments>}}\n\n'
        "The tool is chosen by 'tool' alone; 'kind' is not an argument.\n\n"
        "Available tools:\n" + tool_contract()
    )


def _messages(system: str, observations: list[str], completions: list[str]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}]
    for index, observation in enumerate(observations):
        messages.append({"role": "user", "content": observation})
        if index < len(completions):
            messages.append({"role": "assistant", "content": completions[index]})
    return messages


@torch.no_grad()
def rollout_v2(
    model: Any,
    tokenizer: Any,
    scenario: FrozenScenario,
    *,
    system_prompt: str,
    max_actions: int = protocol_v2.MAX_ACTIONS,
    max_new_tokens: int = protocol_v2.MAX_NEW_TOKENS,
    max_prompt_tokens: int = protocol_v2.MAX_PROMPT_TOKENS,
    temperature: float = protocol_v2.TRAIN_TEMPERATURE,
    device: torch.device | None = None,
    before_generate: Callable[[int], None] | None = None,
) -> EpisodeV2:
    """Run one episode, keeping the exact tokens of every turn.

    ``before_generate`` is called with the step index immediately before each
    ``generate``. The v2 pilot passes a deadline check here: a generation is the
    longest single operation in the run, so a budget that is only consulted
    between episodes cannot bound it.
    """
    device = device or torch.device("cpu")
    observations: list[str] = []
    completions: list[str] = []
    turns: list[TurnRecord] = []
    counters: dict[str, int] = {c.value: 0 for c in ActionCategory}
    context_exhausted = {"hit": False}
    eos_ids = {tokenizer.eos_token_id, *(tokenizer.additional_special_tokens_ids or [])}

    def choose(step: int, observation: Any) -> Action | None:
        observations.append(render_observation(observation))
        prompt_text = tokenizer.apply_chat_template(
            _messages(system_prompt, observations, completions),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = encoded["input_ids"].to(device)
        if prompt_ids.shape[1] > max_prompt_tokens:
            observations.pop()
            context_exhausted["hit"] = True
            return None

        if before_generate is not None:
            before_generate(step)

        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature, **protocol_v2.SAMPLING_NEUTRALISED}
            if temperature > 0
            else {"do_sample": False}
        )
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **sampling,
        )
        generated = out[0, prompt_ids.shape[1] :]
        if generated.numel() == 0:
            raise TokenProvenanceError(
                f"step {step} generated no tokens; a turn with an empty span "
                f"cannot be scored and must not be silently dropped",
            )
        generated_ids = tuple(int(t) for t in generated)
        stop_reason = (
            "eos" if generated_ids[-1] in eos_ids
            else ("length" if len(generated_ids) >= max_new_tokens else "stopped")
        )
        text = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append(text)

        action = parse_action(text)
        category = categorise(action)
        counters[category.value] += 1
        turns.append(
            TurnRecord(
                step_index=step,
                prompt_ids=tuple(int(t) for t in prompt_ids[0]),
                generated_ids=generated_ids,
                stop_reason=stop_reason,
                text=text,
                action_kind=str(action.kind),
                category=category.value,
                outcome="",
            ),
        )
        return action

    env, actions, declared = drive(scenario, choose, max_actions=max_actions)
    recorded = [
        turn.model_copy(update={"outcome": str(entry.outcome)})
        for turn, entry in zip(turns, env.world.trace.agent_entries(), strict=False)
    ]
    if len(recorded) != len(actions):
        raise TokenProvenanceError(
            f"recorded {len(recorded)} turns for {len(actions)} executed actions",
        )

    world = env.world
    verdict = verify(scenario, scenario.world, world, world.trace)
    episode = Episode(
        scenario_id=scenario.scenario_id,
        turns=(),
        actions=tuple(actions),
        reward=episode_reward(verdict),
        task_completion=float(verdict.task_completion),
        safe_completion=bool(verdict.safe_completion),
        decision_correct=bool(verdict.decision_correct),
        committed_violations=tuple(str(v.cost_class) for v in verdict.violations),
        attempted_violations=tuple(str(v.cost_class) for v in verdict.attempted_violations),
        failure_class=str(verdict.failure_class) if verdict.failure_class else None,
        declared_outcome=declared,
        # R6: the verifier's own count, not "every non-malformed action".
        tool_calls=int(verdict.tool_calls),
        malformed_actions=counters[ActionCategory.MALFORMED.value],
        step_limited=declared is None,
        terminal_state_hash=world.state_hash(),
        trace_head_hash=world.trace.head_hash,
    )
    return EpisodeV2(
        turns=tuple(recorded),
        episode=episode,
        # R9: v1 collapsed both exhaustion modes into ``step_limited``. They are
        # different failures -- one is the policy running out of moves, the other
        # is the harness running out of context -- and only one is about the policy.
        termination=(
            "declared" if declared is not None
            else ("context_exhausted" if context_exhausted["hit"] else "action_limit")
        ),
    )
