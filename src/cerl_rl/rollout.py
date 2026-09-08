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
from pydantic import TypeAdapter, ValidationError

from cerl.actions import Action, MalformedAction
from cerl.agents.tool_schemas import all_tool_schemas, tool_name_to_kind
from cerl.core import Frozen
from cerl.env.env import CerlEnv
from cerl.env.render import render_observation
from cerl.env.reward import CostVector, RewardVector, default_scalar
from cerl.scenario.schema import FrozenScenario
from cerl.verify.verifier import verify

_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

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


def tool_menu() -> str:
    lines = []
    for schema in all_tool_schemas():
        props = schema["input_schema"]["properties"]
        args = ", ".join(sorted(props))
        lines.append(f"- {schema['name']}({args})")
    return "\n".join(lines)


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


def parse_action(text: str) -> Action:
    """Model text -> a validated Action, or a scored MalformedAction.

    Exactly the benchmark's contract: an unparseable or invalid turn is a real,
    recorded action that consumes a step. It is never retried, repaired, or
    silently dropped -- doing any of those would make the malformed-turn rate
    unmeasurable, and that rate is the headline finding of the earlier run.
    """
    import json

    snippet = text.strip()
    start, end = snippet.find("{"), snippet.rfind("}")
    if start == -1 or end <= start:
        return MalformedAction(
            raw=snippet[:400],
            parse_error="the turn contained no JSON object; one action per turn is required",
        )
    try:
        payload = json.loads(snippet[start : end + 1])
    except json.JSONDecodeError as error:
        return MalformedAction(raw=snippet[:400], parse_error=f"invalid JSON: {error.msg}")
    if not isinstance(payload, dict) or "tool" not in payload:
        return MalformedAction(raw=snippet[:400], parse_error="no 'tool' key in the object")

    try:
        kind = tool_name_to_kind(str(payload["tool"]))
    except Exception:  # noqa: BLE001 - any unknown name is simply malformed
        return MalformedAction(
            raw=snippet[:400], parse_error=f"unknown tool {payload['tool']!r}",
        )
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        return MalformedAction(raw=snippet[:400], parse_error="'arguments' must be an object")
    try:
        return _ACTION_ADAPTER.validate_python({"kind": kind, **arguments})
    except ValidationError as error:
        return MalformedAction(
            raw=snippet[:400],
            parse_error=f"{kind}: {error.errors()[0].get('msg', 'invalid arguments')}",
        )


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
    """Run one episode. A fresh env every time -- never a reused or reset-in-place one."""
    env = CerlEnv(scenario)          # isolated: constructed from the frozen scenario
    observation = env.reset()        # exact initial state

    observations: list[str] = []
    completions: list[str] = []
    turns: list[Turn] = []
    actions: list[Action] = []
    spans: list[tuple[int, int]] = []   # (start, end) of generated tokens in the final sequence
    malformed = 0
    tool_calls = 0
    declared: str | None = None

    for step in range(max_actions):
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
            # Context exhausted. Recorded as a step-limited episode rather than
            # silently truncating the conversation, which would change what the
            # model saw without saying so.
            break

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
        gen_ids = out[0, prompt_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        spans.append((prompt_ids.shape[1], out.shape[1]))
        completions.append(text)

        action = parse_action(text)
        actions.append(action)
        if str(action.kind) == "malformed":
            malformed += 1
        else:
            tool_calls += 1

        result = env.step(action)
        entry = env.world.trace.entries[-1]
        turns.append(
            Turn(step_index=step, prompt_text=prompt_text, completion_text=text,
                 action_kind=str(action.kind), outcome=str(entry.outcome),
                 malformed=str(action.kind) == "malformed"),
        )
        observation = result.observation
        if result.terminated:
            declared = str(action.kind)
            break

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


def episode_reward(verdict: Any) -> float:
    """The pilot's training objective: ``cerl``'s own ``default_scalar``.

    The formula is not touched. What this does is assemble its two arguments
    for a *finished episode*, because the per-step reward the environment
    returns is deliberately empty -- the terminal components come from the
    verifier, so a mid-episode scalar would always be zero.

    ``r_efficiency`` is left at its default because ``default_scalar`` does not
    weight it. Committed costs enter through ``CostVector``; attempted costs
    cannot, because ``default_scalar`` has no parameter for them. That is the
    Phase-1 decision (CLAUDE.md rule 2) and it is inherited here rather than
    quietly revisited: this pilot does not train against blocked intent.
    """
    committed = CostVector.from_classes(tuple(v.cost_class for v in verdict.violations))
    reward_vector = RewardVector(
        r_outcome=1.0 if verdict.correct_final_state else 0.0,
        r_task=float(verdict.task_completion),
        r_decision=1.0 if verdict.decision_correct else 0.0,
    )
    return float(default_scalar(reward_vector, committed))


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
