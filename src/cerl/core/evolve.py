"""Validated immutable updates.

``BaseModel.model_copy(update=...)`` **bypasses validation entirely**. Whatever
you hand it is installed on the new instance verbatim: a raw ``dict`` where a
``FrozenMap`` is declared, a ``TicketId`` where a ``CustomerId`` is declared, a
mutable ``list``, an out-of-range enum value. Nothing raises, and the corruption
surfaces later as a wrong diff, an unstable hash, or -- worst of all -- a verdict
that is quietly wrong.

For a benchmark whose entire value rests on state integrity that is not an
acceptable primitive. ``evolve`` validates every changed field against its
declared annotation before installing it, so the invariants the type annotations
claim are actually enforced at every transition.

It is deliberately cheap on the hot path: Pydantic short-circuits validation of a
value that is already an instance of the declared model, so passing an
already-constructed submodel costs a type check, while passing a raw dict pays
for the full validation that makes it safe.

``tests/immutability`` forbids ``model_copy(update=...)`` anywhere in ``state``,
``tools`` or ``env``.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

ModelT = TypeVar("ModelT", bound=BaseModel)

_ADAPTERS: dict[tuple[type[BaseModel], str], TypeAdapter[Any]] = {}


class UnknownField(ValueError):
    """An update named a field the model does not declare."""

    def __init__(self, model: type[BaseModel], field: str) -> None:
        super().__init__(
            f"{model.__name__} has no field {field!r}; "
            f"declared fields are {sorted(model.model_fields)}",
        )
        self.model = model
        self.field = field


def _adapter(model: type[BaseModel], field: str) -> TypeAdapter[Any]:
    key = (model, field)
    cached = _ADAPTERS.get(key)
    if cached is None:
        info = model.model_fields.get(field)
        if info is None:
            raise UnknownField(model, field)
        cached = TypeAdapter(info.annotation)
        _ADAPTERS[key] = cached
    return cached


def validate_field(model: type[BaseModel], field: str, value: Any) -> Any:
    """Validate one value against ``model.field``'s declared annotation."""
    return _adapter(model, field).validate_python(value)


def evolve(model: ModelT, /, **changes: Any) -> ModelT:
    """Return a copy of ``model`` with ``changes`` applied, fully validated.

    Raises ``UnknownField`` for a field the model does not declare, and
    ``ValidationError`` for a value that does not satisfy its annotation.
    Mutable containers are coerced to their immutable declared types rather than
    being installed as-is.
    """
    cls = type(model)
    validated = {
        field: validate_field(cls, field, value) for field, value in changes.items()
    }
    return model.model_copy(update=validated)
