"""Validated nominal entity identifiers.

``NewType("CustomerId", str)`` is erased at runtime: it gives mypy nominal
checking and *zero* runtime validation, so a ``TicketId`` deserialized into a
``CustomerId`` field is accepted silently. Since near-duplicate-entity confusion
is the central hazard of this benchmark, an id mix-up that a frozen scenario
file could carry undetected is exactly the bug most likely to produce a false
research result.

These ``str`` subclasses give all three properties at once:

* **static**  - a bare ``str`` is not assignable to ``CustomerId`` under
  ``mypy --strict``, and a ``TicketId`` where a ``CustomerId`` is expected is an
  error;
* **runtime** - construction *and* Pydantic deserialization reject a wrong
  prefix, so a swapped id in a scenario file fails at load, not in a verdict;
* **serialization** - still a plain JSON string, so canonical JSON and RFC 6901
  pointer paths are unaffected.

See CLAUDE.md rule 8.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic_core import core_schema

from cerl.core.errors import InvalidEntityId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Self

    from pydantic import GetCoreSchemaHandler

_SUFFIX = r"[0-9a-f]{12}"


class EntityId(str):
    """Base for prefixed, format-validated identifiers."""

    __slots__ = ()

    PREFIX: ClassVar[str] = ""
    PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"$^")

    def __new__(cls, raw: str) -> Self:
        if cls is EntityId:
            raise InvalidEntityId("EntityId", raw)
        if not isinstance(raw, str) or not cls.PATTERN.fullmatch(raw):
            raise InvalidEntityId(cls.__name__, str(raw))
        return super().__new__(cls, raw)

    @classmethod
    def mint(cls, index: int) -> Self:
        """Deterministically build the ``index``-th id of this type.

        Used by the scenario generator; the digits come from a keyed derivation
        upstream, never from ``random`` or ``uuid`` (CLAUDE.md rule 1).
        """
        return cls(f"{cls.PREFIX}_{index:012x}")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def _validate(value: str) -> EntityId:
            return cls(value)

        return core_schema.no_info_after_validator_function(
            _validate,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str,
                return_schema=core_schema.str_schema(),
                when_used="always",
            ),
        )


class CustomerId(EntityId):
    PREFIX = "cus"
    PATTERN = re.compile(rf"cus_{_SUFFIX}")


class ChargeId(EntityId):
    PREFIX = "ch"
    PATTERN = re.compile(rf"ch_{_SUFFIX}")


class RefundId(EntityId):
    PREFIX = "re"
    PATTERN = re.compile(rf"re_{_SUFFIX}")


class TicketId(EntityId):
    PREFIX = "tkt"
    PATTERN = re.compile(rf"tkt_{_SUFFIX}")


class UserId(EntityId):
    PREFIX = "u"
    PATTERN = re.compile(rf"u_{_SUFFIX}")


class ApprovalId(EntityId):
    PREFIX = "apr"
    PATTERN = re.compile(rf"apr_{_SUFFIX}")


class MessageId(EntityId):
    PREFIX = "msg"
    PATTERN = re.compile(rf"msg_{_SUFFIX}")


class InvoiceId(EntityId):
    PREFIX = "inv"
    PATTERN = re.compile(rf"inv_{_SUFFIX}")


class PaymentMethodId(EntityId):
    PREFIX = "pm"
    PATTERN = re.compile(rf"pm_{_SUFFIX}")


class DisputeId(EntityId):
    PREFIX = "dp"
    PATTERN = re.compile(rf"dp_{_SUFFIX}")


ALL_ID_TYPES: tuple[type[EntityId], ...] = (
    CustomerId,
    ChargeId,
    RefundId,
    TicketId,
    UserId,
    ApprovalId,
    MessageId,
    InvoiceId,
    PaymentMethodId,
    DisputeId,
)
