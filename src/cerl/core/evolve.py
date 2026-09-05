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

    The changed values are checked against their declared annotations first, so
    a failure names the offending field, and then the **complete reconstructed
    model is validated through its own class**. That second step is not
    redundant: field-level validation alone cannot see model-level
    ``@model_validator`` logic or any invariant spanning two fields, so a
    combination of individually valid values could otherwise produce an invalid
    object -- a charge refunded for more than it was worth, an approval that
    expires before it was granted.

    ``model_copy(update=...)`` is deliberately **not** used as the validation
    boundary anywhere in this function: it installs values unchecked, which is
    the whole reason this helper exists. Constructing through
    ``model_validate`` also yields a genuinely new instance, so any private
    memoised value (such as ``WorldState``'s cached document and state hash) is
    reset rather than inherited stale.

    Raises ``UnknownField`` for an undeclared field and ``ValidationError`` for a
    value -- or a combination of values -- that the model rejects.
    """
    cls = type(model)
    for field in changes:
        if field not in cls.model_fields:
            raise UnknownField(cls, field)

    # Validate each change on its own first: the error then points at the field
    # rather than at the whole model.
    validated = {
        field: validate_field(cls, field, value) for field, value in changes.items()
    }

    merged: dict[str, Any] = {
        name: getattr(model, name) for name in cls.model_fields
    }
    merged.update(validated)
    return cls.model_validate(merged)
