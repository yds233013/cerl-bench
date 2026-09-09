"""Multi-turn rollouts against the real CERL environment.

The environment is not reimplemented or simplified here. Each rollout builds a
fresh :class:`CerlEnv` from the frozen scenario, so it starts from the exact
initial state; actions go through the same validated dispatcher the benchmark
uses, so responders, the logical clock, Layer-C interlocks and the verifier keep
their ordinary semantics. What this module adds is only the parts a trainer
needs: a token-level record of which tokens the model produced, and a reward
read off the finished episode.
"""

from __future__ import annotations

from typing import Any

import torch

from cerl.actions import Action
from cerl.core import Frozen
from cerl.env.render import render_observation
from cerl.scenario.schema import FrozenScenario
from cerl.verify.verifier import verify
from cerl_rl.environment import (
    drive,
    episode_reward,
    parse_action,
    tool_menu,
)

__all__ = [
    "Episode", "TokenisedEpisode", "Turn", "episode_reward", "parse_action",
    "rollout", "tool_menu",
]

#: A 0.6B model cannot be relied on for native tool-call formatting, so the
#: contract is one JSON object per turn. The *parsing* is deliberately no more
#: forgiving than the benchmark's: anything that does not validate becomes a
#: MalformedAction and is scored as one. Being lenient here would inflate the
#: result by hiding exactly the failure the earlier 4B run exhibited.
SYSTEM_SUFFIX = """

Reply with ONE JSON object and nothing else. No prose, no explanation, no
markdown fence. The object has exactly two keys:

  {"tool": "<tool name>", "arguments": {<arguments>}}

Available tools:
%s

Example: {"tool": "tickets__get", "arguments": {"ticket_id": "tkt_000000000001"}}"""


class Turn(Frozen):
    """One assistant turn, with the token span the model actually produced."""

    step_index: int
    prompt_text: str
    completion_text: str
    action_kind: str
    outcome: str
    malformed: bool


class Episode(Frozen):
    """A finished rollout: what happened, and what it is worth."""

    scenario_id: str
    turns: tuple[Turn, ...]
    actions: tuple[Action, ...]
    reward: float
    task_completion: float
    safe_completion: bool
    decision_correct: bool
    committed_violations: tuple[str, ...]
    attempted_violations: tuple[str, ...]
    failure_class: str | None
    declared_outcome: str | None
    tool_calls: int
    malformed_actions: int
    step_limited: bool
    terminal_state_hash: str
    trace_head_hash: str

    @property
    def n_generated_turns(self) -> int:
        return len(self.turns)


class TokenisedEpisode(Frozen, arbitrary_types_allowed=True):
    """The episode as the trainer sees it: one sequence, one mask.

    ``input_ids`` is the whole conversation -- system prompt, observations, tool
    results, responder-visible text and the model's own turns. ``gen_mask`` is 1
    **only** on tokens the model generated. Everything the environment produced
    is 0, so no gradient ever flows through a token the policy did not choose.
    Getting this wrong is the quiet way to train on your own observations and
    report it as learning.
    """

    input_ids: torch.Tensor  # (T,)
    gen_mask: torch.Tensor  # (T,) 1 on model-generated tokens
    episode: Episode

    @property
    def n_generated_tokens(self) -> int:
        return int(self.gen_mask.sum().item())


def _messages(system: str, observations: list[str], completions: list[str]) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system}]
    for i, obs in enumerate(observations):
        msgs.append({"role": "user", "content": obs})
        if i < len(completions):
            msgs.append({"role": "assistant", "content": completions[i]})
    return msgs


@torch.no_grad()
def rollout(
    model: Any,
    tokenizer: Any,
    scenario: FrozenScenario,
    *,
    max_actions: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    system_prompt: str,
    max_prompt_tokens: int,
) -> TokenisedEpisode:
    """Run one episode with the model choosing every action.

    The episode's control flow -- fresh environment, terminal action, step limit
    -- is :func:`cerl_rl.environment.drive`, shared with the offline audit so
    there is one definition of "what an episode is" rather than a training copy
    and a review copy that can drift apart.
    """
    observations: list[str] = []
    completions: list[str] = []
    turns: list[Turn] = []
    counters = {"malformed": 0, "tool_calls": 0}

    def choose(step: int, observation: Any) -> Action | None:
        observations.append(render_observation(observation))
        prompt_text = tokenizer.apply_chat_template(
            _messages(system_prompt, observations, completions),
            tokenize=False,
            add_generation_prompt=True,
            # The template, not a runner flag: this appends a pre-closed
            # <think></think> block so the model starts after it. Verified by
            # rendering the prompt, not assumed.
            enable_thinking=False,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if prompt_ids.shape[1] > max_prompt_tokens:
            # Context exhausted. The episode ends here and is recorded as
            # step-limited, rather than silently dropping earlier turns -- that
            # would change what the model saw without saying so.
            observations.pop()
            return None

        # Sampling arguments are passed only when sampling. Passing them with
        # do_sample=False makes transformers warn that they are ignored, and a
        # warning that is always present is a warning nobody reads.
        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.95}
            if temperature > 0
            else {"do_sample": False}
        )
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **sampling,
        )
        text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        completions.append(text)

        action = parse_action(text)
        malformed = str(action.kind) == "malformed"
        counters["malformed" if malformed else "tool_calls"] += 1
        turns.append(
            Turn(step_index=step, prompt_text=prompt_text, completion_text=text,
                 action_kind=str(action.kind), outcome="", malformed=malformed),
        )
        return action

    env, actions, declared = drive(scenario, choose, max_actions=max_actions)
    # Outcomes are read back from the trace the environment actually wrote.
    turns = [
        turn.model_copy(update={"outcome": str(entry.outcome)})
        for turn, entry in zip(turns, env.world.trace.agent_entries(), strict=False)
    ]
    malformed = counters["malformed"]
    tool_calls = counters["tool_calls"]

    world = env.world
    verdict = verify(scenario, scenario.world, world, world.trace)
    reward = episode_reward(verdict)

    episode = Episode(
        scenario_id=scenario.scenario_id,
        turns=tuple(turns),
        actions=tuple(actions),
        reward=float(reward),
        task_completion=float(verdict.task_completion),
        safe_completion=bool(verdict.safe_completion),
        decision_correct=bool(verdict.decision_correct),
        committed_violations=tuple(str(v.cost_class) for v in verdict.violations),
        attempted_violations=tuple(str(v.cost_class) for v in verdict.attempted_violations),
        failure_class=str(verdict.failure_class) if verdict.failure_class else None,
        declared_outcome=declared,
        tool_calls=tool_calls,
        malformed_actions=malformed,
        step_limited=declared is None,
        terminal_state_hash=world.state_hash(),
        trace_head_hash=world.trace.head_hash,
    )

    # Re-tokenise the finished conversation once, and locate the generated spans
    # inside it. The mask is built from those spans, so it cannot drift from
    # what was actually sampled.
    full_text = tokenizer.apply_chat_template(
        _messages(system_prompt, observations[: len(completions)], completions),
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    mask = _generated_mask(tokenizer, ids, completions)
    return TokenisedEpisode(input_ids=ids, gen_mask=mask, episode=episode)


def _generated_mask(tokenizer: Any, ids: torch.Tensor, completions: list[str]) -> torch.Tensor:
    """1 on tokens belonging to an assistant turn, 0 everywhere else.

    Located by matching each completion's token ids inside the full sequence, so
    the mask follows the text the model actually produced rather than a guess at
    where the template put it.
    """
    mask = torch.zeros_like(ids)
    cursor = 0
    for text in completions:
        if not text:
            continue
        needle = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not needle:
            continue
        found = _find(ids.tolist(), needle, cursor)
        if found is None:
            continue
        mask[found : found + len(needle)] = 1
        cursor = found + len(needle)
    return mask


def _find(hay: list[int], needle: list[int], start: int) -> int | None:
    n = len(needle)
    for i in range(start, len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    return None
