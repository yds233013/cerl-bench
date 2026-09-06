"""The local-model client, tested without a model server.

Every test here uses a fake transport. The default suite must not depend on a
running model: a test that silently passes because `ollama serve` happens to be
up is not a test, and CI has no GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from cerl.agents.local_client import (
    DEFAULT_NUM_CTX,
    LocalContextExhausted,
    LocalModelClient,
    LocalModelInfo,
    LocalRunnerUnavailable,
    NonLocalHost,
    OllamaTransport,
    to_openai_tools,
)
from cerl.agents.tool_schemas import all_tool_schemas


class FakeTransport:
    """Scripted replies in the runner's wire shape. Never touches a network."""

    def __init__(self, replies: list[dict[str, Any]] | None = None) -> None:
        self.replies = replies or []
        self.calls: list[dict[str, Any]] = []
        self.timeouts: list[float | None] = []
        self.index = 0

    def chat(
        self, payload: dict[str, Any], timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(payload)
        self.timeouts.append(timeout_s)
        if self.index < len(self.replies):
            reply = self.replies[self.index]
            self.index += 1
            return reply
        return _reply(content="nothing further")

    def version(self) -> str:
        return "0.0.0-fake"

    def tags(self) -> dict[str, Any]:
        return {"models": []}


def _reply(
    *,
    content: str = "",
    tool: str | None = None,
    arguments: Any = None,
    done_reason: str = "stop",
    prompt_tokens: int | None = 100,
    output_tokens: int | None = 20,
) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content}
    if tool is not None:
        message["tool_calls"] = [
            {"function": {"name": tool, "arguments": arguments or {}}},
        ]
    reply: dict[str, Any] = {"message": message, "done_reason": done_reason}
    if prompt_tokens is not None:
        reply["prompt_eval_count"] = prompt_tokens
    if output_tokens is not None:
        reply["eval_count"] = output_tokens
    return reply


def _client(transport: Any, **kwargs: Any) -> LocalModelClient:
    return LocalModelClient(
        transport=transport,
        info=LocalModelInfo(digest="d" * 64, model="fake:1b"),
        **kwargs,
    )


# --------------------------------------------------------------------------
# local-only, structurally
# --------------------------------------------------------------------------


def test_a_remote_host_is_refused():
    """Local-only must be a property, not a promise."""
    with pytest.raises(NonLocalHost):
        OllamaTransport(host="https://api.example.com")


@pytest.mark.parametrize("host", ["http://127.0.0.1:11434", "http://localhost:11434"])
def test_loopback_hosts_are_accepted(host):
    assert OllamaTransport(host=host).host == host


def test_binding_to_all_interfaces_is_not_local():
    """0.0.0.0 means every interface, which is the opposite of local-only."""
    with pytest.raises(NonLocalHost):
        OllamaTransport(host="http://0.0.0.0:11434")


def test_the_client_carries_no_credential():
    """There is no code path from here to a paid provider."""
    import inspect

    from cerl.agents import local_client

    source = inspect.getsource(local_client)
    for token in ("api_key", "ANTHROPIC", "Authorization", "Bearer"):
        assert token not in source, token


def test_provenance_names_the_actual_model():
    info = LocalModelInfo(model="qwen3:4b", digest="359d7dd4bcda" + "0" * 52)
    assert info.provenance == "local/ollama/qwen3:4b@359d7dd4bcda"


# --------------------------------------------------------------------------
# the ordinary path
# --------------------------------------------------------------------------


def test_a_tool_call_becomes_a_model_response():
    transport = FakeTransport([_reply(tool="tickets__get", arguments={"ticket_id": "t"})])
    response = _client(transport).complete("sys", [], [])
    assert response.tool_name == "tickets__get"
    assert dict(response.tool_input) == {"ticket_id": "t"}
    assert response.source == "local"


def test_the_response_is_never_labelled_live_or_synthetic():
    """Provenance must distinguish a local run from a paid one and from a fixture."""
    transport = FakeTransport([_reply(tool="abstain", arguments={"reason": "x"})])
    assert _client(transport).complete("sys", [], []).source == "local"


def test_string_arguments_are_parsed():
    """Some runners return the arguments object as a JSON string."""
    transport = FakeTransport(
        [_reply(tool="tickets__get", arguments='{"ticket_id": "t"}')],
    )
    response = _client(transport).complete("sys", [], [])
    assert dict(response.tool_input) == {"ticket_id": "t"}


def test_unparseable_string_arguments_are_surfaced_not_swallowed():
    transport = FakeTransport([_reply(tool="tickets__get", arguments="{not json")])
    response = _client(transport).complete("sys", [], [])
    assert "__unparsed__" in dict(response.tool_input)


def test_only_the_first_tool_call_is_taken():
    """One action per turn is the environment's contract.

    Replaying a queued second call later would attribute an action to an
    observation the model never saw.
    """
    transport = FakeTransport(
        [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "tickets__get", "arguments": {}}},
                        {"function": {"name": "abstain", "arguments": {}}},
                    ],
                },
                "done_reason": "stop",
            },
        ],
    )
    assert _client(transport).complete("sys", [], []).tool_name == "tickets__get"


def test_the_system_prompt_and_tools_reach_the_runner():
    transport = FakeTransport([_reply(tool="abstain")])
    tools = list(all_tool_schemas())
    _client(transport).complete("SYSTEM", [{"role": "user", "content": "obs"}], tools)
    payload = transport.calls[0]
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert len(payload["tools"]) == len(tools)
    assert payload["options"]["num_ctx"] == DEFAULT_NUM_CTX


def test_tool_schemas_convert_without_changing_the_schema():
    """Only the envelope differs between conventions; the schema is untouched."""
    tools = list(all_tool_schemas())
    converted = to_openai_tools(tools)
    assert len(converted) == len(tools)
    for original, new in zip(tools, converted, strict=True):
        assert new["type"] == "function"
        assert new["function"]["name"] == original["name"]
        assert new["function"]["parameters"] == original["input_schema"]


# --------------------------------------------------------------------------
# failure handling -- each mode is a real behaviour, not an error to hide
# --------------------------------------------------------------------------


def test_prose_without_a_tool_call_yields_no_tool_name():
    """The agent scores this as a MalformedAction; the client does not retry."""
    transport = FakeTransport([_reply(content="I would refund the duplicate.")])
    response = _client(transport).complete("sys", [], [])
    assert response.tool_name is None
    assert response.text.startswith("I would refund")


def test_output_exhaustion_is_reported_as_such():
    """Generation cut off before the tool call. Truthfully a truncated turn."""
    transport = FakeTransport(
        [_reply(content="thinking..." * 20, done_reason="length")],
    )
    response = _client(transport).complete("sys", [], [])
    assert response.stop_reason == "length"
    assert response.tool_name is None


def test_context_exhaustion_stops_the_episode_rather_than_truncating(caplog):
    """Silently dropping history changes what the model reasons about."""
    transport = FakeTransport([_reply(tool="abstain", prompt_tokens=99_999)])
    with pytest.raises(LocalContextExhausted):
        _client(transport, num_ctx=4096).complete("sys", [], [])


def test_a_runner_error_about_context_is_classified_as_context_exhaustion():
    transport = FakeTransport([{"error": "context length exceeded"}])
    with pytest.raises(LocalContextExhausted):
        _client(transport).complete("sys", [], [])


def test_any_other_runner_error_is_raised_not_retried_elsewhere():
    transport = FakeTransport([{"error": "model not found"}])
    with pytest.raises(LocalRunnerUnavailable, match="model not found"):
        _client(transport).complete("sys", [], [])


def test_an_unreachable_runner_says_so_and_names_no_provider():
    transport = OllamaTransport(host="http://127.0.0.1:1")
    client = LocalModelClient(transport=transport, info=LocalModelInfo())
    with pytest.raises(LocalRunnerUnavailable) as excinfo:
        client.complete("sys", [], [])
    assert "No remote provider" in str(excinfo.value)


def test_an_unknown_tool_is_passed_through_for_the_agent_to_reject():
    """Deciding what is a valid action belongs to the action schema, not here."""
    transport = FakeTransport([_reply(tool="not__a__tool", arguments={"x": 1})])
    assert _client(transport).complete("sys", [], []).tool_name == "not__a__tool"


# --------------------------------------------------------------------------
# usage accounting
# --------------------------------------------------------------------------


def test_usage_totals_tokens_and_runtime():
    transport = FakeTransport(
        [_reply(tool="abstain", prompt_tokens=100, output_tokens=20)] * 3,
    )
    client = _client(transport)
    for _ in range(3):
        client.complete("sys", [], [])
    usage = client.usage()
    assert usage["turns"] == 3
    assert usage["prompt_tokens"] == 300
    assert usage["output_tokens"] == 60
    assert usage["wall_seconds"] >= 0.0


def test_unavailable_token_counts_are_marked_not_zeroed():
    """Reporting an unknown as 0 would read as a measurement."""
    transport = FakeTransport([_reply(tool="abstain", prompt_tokens=None)])
    client = _client(transport)
    client.complete("sys", [], [])
    assert client.usage()["prompt_tokens"] is None
    assert client.usage()["output_tokens"] == 20


def test_truncated_turns_are_counted():
    transport = FakeTransport(
        [_reply(content="x", done_reason="length"), _reply(tool="abstain")],
    )
    client = _client(transport)
    client.complete("sys", [], [])
    client.complete("sys", [], [])
    assert client.usage()["truncated_turns"] == 1


# --------------------------------------------------------------------------
# the thinking-channel contract (diagnosed 2026-09-06)
# --------------------------------------------------------------------------


def test_thinking_is_requested_so_reasoning_does_not_become_content():
    """Regression: we sent think=false and got reasoning in ``content``.

    The chat template primes a ``<think>`` block on every request whose last
    message is not an assistant turn -- every request an agent makes. With
    think=false the runner does not *parse* that block, so the model's reasoning
    arrives as ordinary content. It reasons either way; the flag only decides
    where the text lands.
    """
    transport = FakeTransport([_reply(tool="abstain")])
    _client(transport).complete("sys", [], [])
    assert transport.calls[0]["think"] is True


def test_reasoning_is_never_returned_as_the_turns_content():
    """The agent appends content to the conversation.

    Returning reasoning here fed the model its own preamble every turn, which is
    against Qwen3's guidance and is how prompts grew without bound.
    """
    transport = FakeTransport(
        [
            {
                "message": {
                    "content": "FINAL",
                    "thinking": "Okay, let me think about this at length. " * 40,
                    "tool_calls": [{"function": {"name": "abstain", "arguments": {}}}],
                },
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        ],
    )
    client = _client(transport)
    response = client.complete("sys", [], [])
    assert response.text == "FINAL"
    assert "Okay, let me think" not in response.text
    # recorded for diagnosis, but kept out of the conversation
    assert client.turns[0].thinking_chars > 0


def test_thinking_can_be_disabled_explicitly_for_comparison():
    """Single-factor comparisons need the old behaviour to stay reachable."""
    transport = FakeTransport([_reply(tool="abstain")])
    _client(transport, think=False).complete("sys", [], [])
    assert transport.calls[0]["think"] is False


def test_the_think_setting_is_recorded_in_provenance():
    client = LocalModelClient(transport=FakeTransport(), think=False)
    assert client.info().think is False


def test_timing_is_split_into_prompt_and_generation():
    """Prompt processing and generation are separate costs and scale differently."""
    transport = FakeTransport(
        [
            {
                "message": {"content": "", "tool_calls": [
                    {"function": {"name": "abstain", "arguments": {}}}]},
                "done_reason": "stop",
                "prompt_eval_count": 900,
                "eval_count": 100,
                "prompt_eval_duration": 3_000_000_000,
                "eval_duration": 17_000_000_000,
            },
        ],
    )
    client = _client(transport)
    client.complete("sys", [], [])
    usage = client.usage()
    assert usage["prompt_eval_ms"] == 3000
    assert usage["eval_ms"] == 17000


def test_a_completed_turn_without_a_tool_call_is_not_a_truncated_one():
    """Two different failures that were being reported as one number."""
    transport = FakeTransport(
        [
            _reply(content="I think we should escalate.", done_reason="stop"),
            _reply(content="still reasoning", done_reason="length"),
        ],
    )
    client = _client(transport)
    client.complete("sys", [], [])
    client.complete("sys", [], [])
    assert client.usage()["truncated_turns"] == 1
    assert client.usage()["turns"] == 2
