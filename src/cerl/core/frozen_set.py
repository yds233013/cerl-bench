"""An immutable set that always serializes in sorted order.

A plain ``frozenset`` field serializes in Python's set iteration order, which
varies between processes because string hashing is salted per process. That makes
two runs of the same scenario produce different JSON -- and therefore different
content hashes -- for identical state. It is exactly the class of bug CLAUDE.md
rule 1 forbids ("no unordered iteration in anything serialized"), and it is
invisible until you compare across processes.

``SortedFrozenSet`` keeps set semantics and immutability while serializing as a
sorted list, so the encoding is a function of the *contents* alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_core import core_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import GetCoreSchemaHandler

T = TypeVar("T")


class SortedFrozenSet(frozenset[T]):
    """A frozenset whose JSON form is a sorted list."""

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        args = getattr(source_type, "__args__", None)
        item_schema = handler.generate_schema(args[0]) if args else core_schema.any_schema()
        inner = core_schema.frozenset_schema(item_schema)
        list_schema = core_schema.list_schema(item_schema)
        return core_schema.no_info_after_validator_function(
            cls,
            inner,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: sorted(value, key=str),
                return_schema=list_schema,
                when_used="always",
            ),
        )
