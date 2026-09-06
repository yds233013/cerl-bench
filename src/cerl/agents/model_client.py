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

from cerl.agents.budget import (
    BudgetExceeded,
    SpendLedger,
    TokenCounter,
    resolve_counter,
)
from cerl.core import ExternalInterruption, Frozen, FrozenMap, content_hash

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


class Transport(Protocol):
    """The thing that actually issues a request.

    Extracted so the whole client -- reservation, retry, confirmation, unresolved
    accounting -- can be exercised offline against a synthetic transport. A
    transport declares its own provenance through ``source``, and the client
    stamps every response with it, so a synthetic transport can never produce a
    response labelled ``live``.
    """

    #: "live" or "synthetic". Never blank.
    source: str

    def create(self, **kwargs: Any) -> Any: ...

    def count_tokens(self, **kwargs: Any) -> Any: ...


class AnthropicTransport:
    """The real SDK. Constructed with retries disabled -- see below."""

    source = "live"

    def __init__(self, max_retries: int = 0, timeout: float = 120.0) -> None:
        import anthropic

        # The SDK retries automatically by default. An invisible retry is a
        # billable request the ledger never reserved for, so it is switched off
        # and retrying is done here, where each attempt reserves.
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)

    @property
    def raw(self) -> Any:
        return self._client

    def create(self, **kwargs: Any) -> Any:
        return self._client.messages.create(**kwargs)

    def count_tokens(self, **kwargs: Any) -> Any:
        return self._client.messages.count_tokens(**kwargs)


class AmbiguousRequestOutcome(ExternalInterruption):
    """A request may have reached the provider; its cost is unresolved."""


class AnthropicClient:
    """Model client metered by a :class:`SpendLedger`.

    The authorisation check in ``__init__`` is a *gate*, evaluated once before
    anything is spent. Control comes from the ledger: every request -- including
    every retry -- is reserved against the spendable balance before it is sent,
    and a request whose outcome is ambiguous is charged permanently rather than
    forgiven.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        effort: str = "medium",
        *,
        ledger: SpendLedger | None = None,
        transport: Transport | None = None,
        counter: TokenCounter | None = None,
        max_retries: int = 2,
        require_authorization: bool = True,
    ) -> None:
        if require_authorization and not live_evaluation_authorized():
            raise LiveEvaluationNotAuthorized(
                f"live evaluation requires {AUTHORIZATION_ENV}=1 and a positive "
                f"{BUDGET_ENV}. Credentials alone are not a spending budget.",
            )
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_retries = max_retries
        self._transport = transport
        self._counter = counter

        if ledger is None:
            ledger = SpendLedger(
                cap_cents=float(live_evaluation_budget()), model=model,
            )
        elif require_authorization and ledger.cap_cents > live_evaluation_budget():
            raise LiveEvaluationNotAuthorized(
                f"ledger cap {ledger.cap_cents}c exceeds the authorised "
                f"{live_evaluation_budget()}c from {BUDGET_ENV}",
            )
        self.ledger = ledger

    # -- lazily built collaborators --------------------------------------

    def _ensure_transport(self) -> Transport:
        if self._transport is None:
            self._transport = AnthropicTransport()
        return self._transport

    def _ensure_counter(self) -> TokenCounter:
        if self._counter is None:
            transport = self._ensure_transport()
            raw = getattr(transport, "raw", None)
            self._counter = resolve_counter(raw, self.model)
            self.ledger.counter_name = self._counter.name
        return self._counter

    # -- one request ------------------------------------------------------

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        transport = self._ensure_transport()
        counter = self._ensure_counter()
        self.ledger.counter_name = counter.name

        message_list = list(messages)
        tool_list = list(tools)
        # The whole request: system prompt, every prior turn, and the tool
        # schemas. Counting only the newest message would under-reserve by the
        # size of the transcript, which dominates a long episode.
        input_tokens = counter.count(system, message_list, tool_list)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            reservation = self.ledger.reserve(
                input_tokens, self.max_tokens, margin=counter.uncertainty_margin,
            )
            try:
                response = transport.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=message_list,
                    tools=tool_list,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                )
            except BudgetExceeded:
                raise
            except Exception as error:
                # The request was transmitted, or may have been. We cannot tell
                # a pre-flight failure from a provider-side one at this level,
                # so the conservative reading is the only safe one: assume we
                # were billed and keep the charge.
                self.ledger.mark_unresolved(
                    reservation, f"{type(error).__name__} on attempt {attempt + 1}",
                )
                last_error = error
                if attempt == self.max_retries:
                    raise AmbiguousRequestOutcome(
                        f"request failed after {attempt + 1} attempts; "
                        f"{self.ledger.unresolved_cents:.3f}c is unresolved",
                    ) from error
                continue

            usage = getattr(response, "usage", None)
            if usage is None or getattr(usage, "output_tokens", None) is None:
                # A response arrived but we cannot price it. It was certainly
                # billed, so the reservation is kept rather than released.
                self.ledger.mark_unresolved(reservation, "response carried no usage")
            else:
                self.ledger.confirm(
                    reservation,
                    int(getattr(usage, "input_tokens", input_tokens) or 0),
                    int(usage.output_tokens or 0),
                    cache_read_tokens=int(
                        getattr(usage, "cache_read_input_tokens", 0) or 0,
                    ),
                    cache_write_tokens=int(
                        getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    ),
                )
            return self._to_response(response, transport.source)

        raise AmbiguousRequestOutcome(f"unreachable: retries exhausted ({last_error})")

    def _to_response(self, response: Any, source: str) -> ModelResponse:
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
            # Provenance comes from the transport, never from a constant. A
            # synthetic transport cannot produce a "live" label.
            source=source,
        )
