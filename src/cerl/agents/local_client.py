"""A local-model client: an open-weight model served by Ollama on this machine.

Local-only by construction. It speaks HTTP to a loopback address and holds no
credential of any kind, so there is no path by which it could reach a paid
provider -- not as a fallback, not on error, not on a malformed reply. A failed
turn stays failed and is scored as the agent's behaviour.

The model is unprivileged in exactly the way every evaluated agent is: it is
handed the rendered observation and the public tool schemas by
``PromptOnlyAgent`` and nothing else. No branch label, no expected action, no
verifier internals, no ground truth. That property is not re-implemented here --
it follows from reusing the existing agent, which structurally cannot receive
those things.

Failure modes are handled explicitly rather than smoothed over, because each one
is a real behaviour the benchmark is meant to measure:

* **no tool call** -- the model wrote prose instead of acting. Returned with
  ``tool_name=None``; the agent turns it into a scored ``MalformedAction``.
* **output exhaustion** -- generation hit ``num_predict`` before the tool call
  was emitted. Reported as ``stop_reason="length"``; also a MalformedAction.
* **context exhaustion** -- the transcript no longer fits. Raised as
  :class:`LocalContextExhausted` so the run stops honestly instead of silently
  truncating the history the model is reasoning from.
* **unknown tool / invalid arguments** -- passed through untouched. The agent's
  existing parser rejects them, which is where that judgement belongs.
* **timeout / server down** -- raised. Never retried into a different provider.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from cerl.agents.model_client import ModelResponse
from cerl.core import Frozen, FrozenMap
from cerl.eval.latency import measure

#: Loopback only. A non-local host would make "local-only" a promise rather than
#: a property, so the client refuses one.
DEFAULT_HOST = "http://127.0.0.1:11434"
#: Loopback names only. ``0.0.0.0`` is deliberately absent: it means "every
#: interface", which is the opposite of local-only.
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")

DEFAULT_MODEL = "qwen3:4b"
#: Enough headroom for a tool call after the model's preamble. Measured: Qwen3
#: writes 250-400 tokens of reasoning prose before acting, and a cap of 200
#: truncates mid-thought and yields no tool call at all.
DEFAULT_NUM_PREDICT = 640
DEFAULT_NUM_CTX = 16384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_S = 300.0

#: Ask the runner to parse the model's reasoning into its own field.
#:
#: **This does not decide whether the model reasons.** Measured on qwen3:4b at
#: this endpoint: the chat template primes an ``<think>`` block on every request
#: whose last message is not an assistant turn, which is every request an agent
#: makes. With ``think=false`` the runner does not parse that block, so the
#: reasoning arrives as ordinary ``content`` -- and the token cost is identical
#: either way (196 vs 196 output tokens on a trivial prompt; 447 vs 447 with a
#: tool). What ``think=true`` changes is *where the reasoning goes*: ``content``
#: becomes the final answer alone and the reasoning lands in ``thinking``.
#:
#: That matters because the agent appends ``content`` to the conversation as its
#: assistant turn. Under ``think=false`` every turn appended a full reasoning
#: transcript to the history, so prompts grew with prose the model then had to
#: re-read. Qwen3's own guidance is not to feed reasoning back into history.
DEFAULT_THINK = True


class LocalRunnerUnavailable(RuntimeError):
    """The local server is not reachable. Never a reason to call a provider."""


class LocalContextExhausted(RuntimeError):
    """The conversation no longer fits in the model's context window."""


class NonLocalHost(ValueError):
    """A host outside loopback was configured for a local-only client."""


class LocalModelInfo(Frozen):
    """Exact provenance of the model that produced a run."""

    runner: str = "ollama"
    runner_version: str = ""
    model: str = DEFAULT_MODEL
    digest: str = ""
    quantization: str = ""
    parameter_size: str = ""
    context_limit: int = 0
    num_ctx: int = DEFAULT_NUM_CTX
    num_predict: int = DEFAULT_NUM_PREDICT
    temperature: float = DEFAULT_TEMPERATURE
    think: bool = DEFAULT_THINK
    license: str = ""

    @property
    def provenance(self) -> str:
        """Short, unambiguous label for reports and transcripts."""
        return f"local/{self.runner}/{self.model}@{self.digest[:12]}"


class LocalTurnStats(Frozen):
    """Per-turn cost, in the units a local run actually has: tokens and seconds."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    wall_seconds: float = 0.0
    done_reason: str = ""
    #: Characters of reasoning. Kept for diagnosis only -- never returned as the
    #: turn's content and never appended to the conversation.
    thinking_chars: int = 0
    prompt_eval_ms: int = 0
    eval_ms: int = 0


def to_openai_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the project's tool schemas to the function-calling shape.

    The schemas themselves are unchanged -- only the envelope differs between
    conventions -- so the model sees exactly the tools every other agent sees.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in tools
    ]


def message_of(response: dict[str, Any]) -> dict[str, Any]:
    return dict(response.get("message", {}) or {})


def _require_local(host: str) -> str:
    stripped = host.split("://", 1)[-1].split("/", 1)[0].split(":", maxsplit=1)[0]
    if stripped not in LOCAL_HOSTS:
        raise NonLocalHost(
            f"local model host must be loopback, got {host!r}. This client is "
            f"local-only by construction and will not reach a remote endpoint.",
        )
    return host


class OllamaTransport:
    """Minimal HTTP transport. Standard library only, so tests need no server."""

    def __init__(self, host: str = DEFAULT_HOST, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.host = _require_local(host)
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 - loopback, checked above
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                return dict(json.load(response))
        except urllib.error.URLError as error:
            raise LocalRunnerUnavailable(
                f"cannot reach the local model runner at {self.host}: {error}. "
                f"Start it with `ollama serve`. No remote provider will be used.",
            ) from error

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.host}{path}", timeout=30) as response:  # noqa: S310
                return dict(json.load(response))
        except urllib.error.URLError as error:
            raise LocalRunnerUnavailable(
                f"cannot reach the local model runner at {self.host}: {error}",
            ) from error

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/chat", payload)

    def version(self) -> str:
        return str(self._get("/api/version").get("version", ""))

    def show(self, model: str) -> dict[str, Any]:
        return self._post("/api/show", {"model": model})

    def tags(self) -> dict[str, Any]:
        return self._get("/api/tags")


class LocalModelClient:
    """``ModelClient`` backed by a locally served open-weight model."""

    name = "local-ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        transport: OllamaTransport | None = None,
        *,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        temperature: float = DEFAULT_TEMPERATURE,
        think: bool = DEFAULT_THINK,
        info: LocalModelInfo | None = None,
    ) -> None:
        self.model = model
        self.transport = transport or OllamaTransport()
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self.think = think
        self._info = info
        self.turns: list[LocalTurnStats] = []

    # -- provenance -------------------------------------------------------

    def info(self) -> LocalModelInfo:
        """Interrogate the runner for exactly what is loaded.

        Read from the runner rather than assumed from a constant: a report that
        names a model it did not actually run is worse than no report.
        """
        if self._info is not None:
            return self._info
        tags = self.transport.tags()
        empty: dict[str, Any] = {}
        entry: dict[str, Any] = next(
            (m for m in tags.get("models", []) if m.get("name") == self.model), empty,
        )
        details = entry.get("details", {})
        self._info = LocalModelInfo(
            runner_version=self.transport.version(),
            model=self.model,
            digest=str(entry.get("digest", "")),
            quantization=str(details.get("quantization_level", "")),
            parameter_size=str(details.get("parameter_size", "")),
            context_limit=int(details.get("context_length", 0) or 0),
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            temperature=self.temperature,
            think=self.think,
        )
        return self._info

    @property
    def provenance(self) -> str:
        return self.info().provenance

    # -- one turn ---------------------------------------------------------

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "think": self.think,
            "messages": [
                {"role": "system", "content": system},
                *[dict(m) for m in messages],
            ],
            "tools": to_openai_tools(tools),
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }
        with measure() as elapsed:
            response = self.transport.chat(payload)

        if "error" in response:
            detail = str(response["error"])
            if "context" in detail.lower() or "too long" in detail.lower():
                raise LocalContextExhausted(
                    f"the conversation no longer fits in {self.num_ctx} tokens: "
                    f"{detail}",
                )
            raise LocalRunnerUnavailable(f"local runner error: {detail}")

        prompt_tokens = response.get("prompt_eval_count")
        output_tokens = response.get("eval_count")
        done_reason = str(response.get("done_reason", ""))

        # The prompt overflowing the window is a real limit being hit, not a
        # model failure, and silently truncating the history would change what
        # the model is reasoning about without saying so.
        if prompt_tokens is not None and int(prompt_tokens) >= self.num_ctx:
            raise LocalContextExhausted(
                f"prompt is {prompt_tokens} tokens against a {self.num_ctx}-token "
                f"window; the episode cannot continue honestly",
            )

        self.turns.append(
            LocalTurnStats(
                prompt_tokens=None if prompt_tokens is None else int(prompt_tokens),
                output_tokens=None if output_tokens is None else int(output_tokens),
                wall_seconds=elapsed.seconds,
                done_reason=done_reason,
                thinking_chars=len(str(message_of(response).get("thinking") or "")),
                prompt_eval_ms=round(int(response.get("prompt_eval_duration", 0)) / 1e6),
                eval_ms=round(int(response.get("eval_duration", 0)) / 1e6),
            ),
        )

        message = message_of(response)
        # Only the final answer. The reasoning stays in ``thinking`` and is
        # deliberately not returned: the agent appends this string to the
        # conversation, and feeding a model its own reasoning back is both
        # against Qwen3's guidance and how prompts grew unboundedly here.
        text = str(message.get("content", "") or "")
        calls = message.get("tool_calls") or []

        tool_name: str | None = None
        tool_input: dict[str, Any] = {}
        if calls:
            # One action per turn is the environment's contract. Extra calls are
            # dropped rather than queued: replaying them later would attribute
            # actions to observations the model never saw.
            function = calls[0].get("function", {})
            tool_name = str(function.get("name", "")) or None
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"__unparsed__": arguments}
            if isinstance(arguments, dict):
                tool_input = dict(arguments)

        return ModelResponse(
            text=text,
            tool_name=tool_name,
            tool_input=FrozenMap(tool_input),
            # "length" means generation was cut off before it finished; the
            # agent scores the turn as malformed, which is the honest reading.
            stop_reason=done_reason or "stop",
            source="local",
        )

    # -- aggregate --------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        """Token and runtime totals, with unavailable values marked, not zeroed."""
        prompt = [t.prompt_tokens for t in self.turns]
        output = [t.output_tokens for t in self.turns]
        return {
            "turns": len(self.turns),
            "prompt_tokens": None if any(p is None for p in prompt) else sum(
                p for p in prompt if p is not None
            ),
            "output_tokens": None if any(o is None for o in output) else sum(
                o for o in output if o is not None
            ),
            "wall_seconds": round(sum(t.wall_seconds for t in self.turns), 2),
            "prompt_eval_ms": sum(t.prompt_eval_ms for t in self.turns),
            "eval_ms": sum(t.eval_ms for t in self.turns),
            "thinking_chars": sum(t.thinking_chars for t in self.turns),
            # Kept apart from "no tool call": a truncated turn ran out of output
            # budget, a completed one chose not to act. Different problems.
            "truncated_turns": sum(1 for t in self.turns if t.done_reason == "length"),
        }
