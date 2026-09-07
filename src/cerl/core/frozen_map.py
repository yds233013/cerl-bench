"""An immutable, hashable ``Mapping`` with a Pydantic core schema.

Pydantic's ``frozen=True`` is *shallow*: it blocks attribute assignment but a
nested ``dict`` field stays mutable in place, so a supposedly frozen snapshot
could be edited through ``state.billing.charges["ch_1"].metadata["x"] = "y"``,
silently corrupting diffs, hashes and reset. ``FrozenMap`` closes that hole for
mapping fields; ``tuple`` and ``frozenset`` cover the others.

See CLAUDE.md rule 8.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_core import core_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import GetCoreSchemaHandler

K = TypeVar("K")
V = TypeVar("V")


class FrozenMap(Mapping[K, V], Hashable):
    """A hashable, immutable mapping preserving canonical (sorted) key order."""

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[K, V] | None = None, /) -> None:
        items = dict(data) if data is not None else {}
        # Sorted by string form of the key so iteration order is canonical and
        # independent of insertion order; keys in this project are always str
        # or str subclasses (entity ids).
        self._data: dict[K, V] = {k: items[k] for k in sorted(items, key=str)}
        self._hash: int | None = None

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMap):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(tuple(self._data.items()))
        return self._hash

    def set(self, key: K, value: V) -> FrozenMap[K, V]:
        """Return a new map with ``key`` bound to ``value``."""
        return FrozenMap({**self._data, key: value})

    def remove(self, key: K) -> FrozenMap[K, V]:
        """Return a new map without ``key``."""
        data = dict(self._data)
        del data[key]
        return FrozenMap(data)

    def to_dict(self) -> dict[K, V]:
        """Return a plain mutable copy (for serialization boundaries only)."""
        return dict(self._data)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Validate from any mapping; serialize as a plain JSON object."""
        args = getattr(source_type, "__args__", None)
        if args and len(args) == 2:  # noqa: PLR2004
            key_schema = handler.generate_schema(args[0])
            value_schema = handler.generate_schema(args[1])
        else:
            key_schema = core_schema.any_schema()
            value_schema = core_schema.any_schema()

        inner = core_schema.dict_schema(key_schema, value_schema)
        return core_schema.no_info_after_validator_function(
            cls,
            inner,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.to_dict(),
                return_schema=inner,
                when_used="always",
            ),
        )


def deep_freeze(value: Any) -> Any:
    """Recursively convert plain JSON containers into immutable ones.

    ``FrozenMap`` freezes only its outer level, so a nested ``dict`` or
    ``list`` inside a ``FrozenMap[str, Any]`` stayed mutable. A tool result is
    handed to the agent *and* sealed into the trace as the same object, so
    editing a nested value through the observation changed the sealed entry and
    broke its hash -- reachable through the ordinary public API, with no
    privileged access at all.

    Mappings become ``FrozenMap`` and sequences become tuples. Both still
    serialise to the same JSON, so canonical encodings and historical replay
    hashes are unchanged.
    """
    # No early return for ``FrozenMap``: its outer level is frozen but its
    # *values* may not be, which is the whole defect this closes.
    if isinstance(value, Mapping):
        return FrozenMap({str(k): deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    return value

def json_copy(value: Any) -> Any:
    """Deep-copy a canonical-JSON document. Faster than ``copy.deepcopy``.

    State documents are pure JSON -- dicts, lists and scalars, no cycles, no
    custom classes -- so ``deepcopy``'s memo table, ``id()`` bookkeeping and
    reductor dispatch are all paid for and none of it is needed. This copier
    recurses over exactly the three container shapes a canonical document can
    hold and returns scalars unchanged, which is safe because every scalar in a
    canonical document is immutable.

    It exists because the copy is on the per-step path: isolating the world
    document costs a full copy on every diff, and ``deepcopy`` made an episode
    exceed the 50 ms budget.
    """
    if isinstance(value, dict):
        return {k: json_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_copy(v) for v in value]
    return value
