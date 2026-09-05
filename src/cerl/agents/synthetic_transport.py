"""A synthetic transport that mimics the provider wire shape, offline.

This exists so the *entire* live path -- token counting, reservation, retry,
confirmation, unresolved accounting, transcript recording, manifest writing --
can be exercised without a paid call. It reproduces the response shape the
client reads, including ``usage``, and can be scripted to fail, to time out, and
to return a response with no usage at all.

**Its ``source`` is ``"synthetic"`` and there is no way to change it.** Every
response the client builds from this transport is stamped synthetic, every
transcript recorded from it is stamped synthetic, and the manifest carries the
transport name. A fixture cannot become a model result by accident, because the
label does not come from a flag anyone can flip -- it comes from the object that
produced the bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SyntheticUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class SyntheticBlock:
    type: str
    text: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticMessage:
    content: tuple[SyntheticBlock, ...]
    stop_reason: str = "tool_use"
    usage: SyntheticUsage | None = None


def tool_turn(
    name: str,
    arguments: dict[str, Any],
    *,
    text: str = "",
    input_tokens: int = 1200,
    output_tokens: int = 180,
) -> SyntheticMessage:
    """A turn that calls one tool, as the agent contract requires."""
    blocks: list[SyntheticBlock] = []
    if text:
        blocks.append(SyntheticBlock(type="text", text=text))
    blocks.append(SyntheticBlock(type="tool_use", name=name, input=dict(arguments)))
    return SyntheticMessage(
        content=tuple(blocks),
        stop_reason="tool_use",
        usage=SyntheticUsage(input_tokens, output_tokens),
    )


def text_turn(
    text: str, *, input_tokens: int = 1200, output_tokens: int = 60,
) -> SyntheticMessage:
    """A turn that names no tool -- which the agent scores as malformed."""
    return SyntheticMessage(
        content=(SyntheticBlock(type="text", text=text),),
        stop_reason="end_turn",
        usage=SyntheticUsage(input_tokens, output_tokens),
    )


def usage_less_turn(name: str, arguments: dict[str, Any]) -> SyntheticMessage:
    """A response that arrived but cannot be priced.

    Certainly billed, unpriceable by us: the ledger must keep the charge.
    """
    return SyntheticMessage(
        content=(SyntheticBlock(type="tool_use", name=name, input=dict(arguments)),),
        stop_reason="tool_use",
        usage=None,
    )


class SyntheticFailure(RuntimeError):
    """A scripted transport failure, standing in for a network or 5xx error."""


class SyntheticTimeout(SyntheticFailure):
    """A scripted timeout: the request may well have reached the provider."""


#: A scripted step is either a message to return or an exception to raise.
Step = SyntheticMessage | BaseException | Callable[[int], SyntheticMessage]


class SyntheticTransport:
    """Returns scripted turns. Cannot be labelled live."""

    source = "synthetic"
    name = "synthetic"

    def __init__(
        self,
        steps: Sequence[Step] = (),
        *,
        default: SyntheticMessage | None = None,
        count_tokens_result: int = 1500,
    ) -> None:
        self._steps = list(steps)
        self._default = default
        self._count = count_tokens_result
        self.requests: list[dict[str, Any]] = []
        self.count_requests: list[dict[str, Any]] = []
        self.index = 0

    @property
    def raw(self) -> None:
        """No underlying SDK client, so no provider token counter is resolved."""
        return None

    def create(self, **kwargs: Any) -> SyntheticMessage:
        self.requests.append(dict(kwargs))
        step: Step
        if self.index < len(self._steps):
            step = self._steps[self.index]
        elif self._default is not None:
            step = self._default
        else:
            step = text_turn("no further scripted turn")
        self.index += 1
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(self.index - 1)
        return step

    def count_tokens(self, **kwargs: Any) -> Any:
        self.count_requests.append(dict(kwargs))

        class _Result:
            input_tokens = self._count

        return _Result()
