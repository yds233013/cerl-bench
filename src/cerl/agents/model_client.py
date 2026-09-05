"""Model clients for the prompt-only baseline.

Three implementations, deliberately separate because they support different
claims:

* ``TranscriptCacheClient`` replays recorded request/response pairs. Offline and
  deterministic. This is what makes a *model* run reproducible, and it is a
  weaker and distinct claim from the environment's offline replay -- the two are
  tested independently.
* ``ScriptedClient`` returns fixed responses for integration testing. Its
  transcripts are stamped ``synthetic`` so they can never be mistaken for, or
  reported as, real model output.
* ``AnthropicClient`` calls the live API. **It refuses to run unless a spending
  budget has been explicitly authorised.** Holding credentials is not the same
  as being authorised to spend, and conflating them is how an unattended run
  bills someone who never agreed to it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

from cerl.core import Frozen, FrozenMap, content_hash

#: Set to a positive integer to authorise live API spend, in whole US cents.
#: Absent or zero means live evaluation is not authorised.
BUDGET_ENV = "CERL_LIVE_EVAL_BUDGET_CENTS"
#: Must be set to "1" alongside a budget. Two independent signals, so a stray
#: budget value in a shell profile cannot by itself start spending.
AUTHORIZATION_ENV = "CERL_LIVE_EVAL_AUTHORIZED"

DEFAULT_MODEL = "claude-opus-5"


class LiveEvaluationNotAuthorized(RuntimeError):
    """Raised when a live run is attempted without an authorised budget."""


class ModelResponse(Frozen):
    """A single model turn, reduced to what the agent needs."""

    text: str = ""
    tool_name: str | None = None
    tool_input: FrozenMap[str, Any] = FrozenMap()
    stop_reason: str = "end_turn"
    #: Provenance. "synthetic" means a fixture and must never be reported as a
    #: model result.
    source: str = "synthetic"


class ModelClient(Protocol):
    """Anything that can turn a conversation into one model turn."""

    name: str

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse: ...


def request_key(
    system: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
) -> str:
    """Content-addressed key for a request, used by the transcript cache."""
    return content_hash(
        {"system": system, "messages": list(messages), "tools": list(tools)},
    )


def live_evaluation_budget() -> int:
    """Authorised spend in whole cents. Zero means not authorised.

    Both signals are required. ``ANTHROPIC_API_KEY`` deliberately plays no part:
    a key proves you *can* call the API, not that anyone agreed to be billed.
    """
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        return 0
    raw = os.environ.get(BUDGET_ENV, "0")
    try:
        budget = int(raw)
    except ValueError:
        return 0
    return max(0, budget)


def live_evaluation_authorized() -> bool:
    return live_evaluation_budget() > 0


class ScriptedClient:
    """Returns pre-set responses. Clearly synthetic; for integration tests only."""

    name = "scripted"

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        self.requests.append(
            {"system": system, "messages": list(messages), "tools": list(tools)},
        )
        if self._index >= len(self._responses):
            return ModelResponse(
                text="no further scripted response", stop_reason="end_turn",
                source="synthetic",
            )
        response = self._responses[self._index]
        self._index += 1
        return response.model_copy(update={"source": "synthetic"})


class TranscriptCacheMiss(KeyError):
    """A cached run was asked for a request the cache does not contain."""


class TranscriptCacheClient:
    """Replays recorded request/response pairs. Offline and deterministic.

    A miss is a hard failure rather than a silent fall-through to a live call:
    a cache that quietly reaches the network is not a reproducibility mechanism.
    """

    name = "transcript-cache"

    def __init__(self, entries: dict[str, ModelResponse]) -> None:
        self._entries = dict(entries)
        self.hits = 0

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        key = request_key(system, messages, tools)
        if key not in self._entries:
            raise TranscriptCacheMiss(
                f"no cached response for request {key[:16]}; the cache cannot be "
                f"completed offline and will not fall back to a live call",
            )
        self.hits += 1
        return self._entries[key]


class AnthropicClient:
    """Live Anthropic API client. Refuses to run without an authorised budget."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        effort: str = "medium",
    ) -> None:
        if not live_evaluation_authorized():
            raise LiveEvaluationNotAuthorized(
                f"live evaluation requires {AUTHORIZATION_ENV}=1 and a positive "
                f"{BUDGET_ENV}. Credentials alone are not a spending budget.",
            )
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        client = self._ensure_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=list(messages),
            tools=list(tools),
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        text = ""
        tool_name = None
        tool_input: dict[str, Any] = {}
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = dict(block.input)
        return ModelResponse(
            text=text,
            tool_name=tool_name,
            tool_input=FrozenMap(tool_input),
            stop_reason=str(response.stop_reason),
            source="live",
        )
