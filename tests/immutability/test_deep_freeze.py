"""Layer 2 gate - deep immutability.

Pydantic's ``frozen=True`` is *shallow*: it blocks attribute assignment but a
nested ``list`` or ``dict`` field stays mutable in place, so a supposedly frozen
snapshot could be edited through it -- corrupting diffs, hashes and reset.

Three independent defences, tested here:

1. an **annotation audit** that fails the moment a mutable container field is
   written, rather than when it causes a bug;
2. a **construction check** that models really do reject assignment;
3. a **behavioural aliasing check** that a full episode leaves the initial
   snapshot's hash untouched (see tests/determinism for the episode-level one).
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

import cerl.actions as actions_pkg
import cerl.state as state_pkg
import cerl.trace as trace_pkg
from cerl.core import Frozen, FrozenMap
from cerl.diff import DiffOp, StateDiff
from cerl.state import WorldState

MUTABLE_ORIGINS = (list, dict, set, frozenset)
FORBIDDEN = (list, dict, set)

# ``Any`` is permitted only where the value is opaque JSON carried through the
# system: diff payloads and FrozenMap value parameters. FrozenMap itself is
# immutable, and DiffOp payloads are never mutated -- they are compared and
# serialised only.
ANY_EXEMPT: frozenset[tuple[str, str]] = frozenset(
    {("DiffOp", "before"), ("DiffOp", "after")},
)


def _models() -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = set()
    for module in (state_pkg, actions_pkg, trace_pkg):
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                found.add(obj)
    found.update({DiffOp, StateDiff, WorldState})
    return found


def _violations(annotation: Any, model_name: str, field_name: str) -> list[str]:
    """Return a message for each mutable container found at any depth."""
    problems: list[str] = []
    origin = get_origin(annotation)

    if annotation in FORBIDDEN or origin in FORBIDDEN:
        problems.append(
            f"{model_name}.{field_name}: mutable container "
            f"{getattr(origin or annotation, '__name__', annotation)!r}",
        )

    if annotation is Any and (model_name, field_name) not in ANY_EXEMPT:
        # Any hides whatever it holds; only the documented payload fields may.
        problems.append(f"{model_name}.{field_name}: bare Any is not an exempt payload field")

    for arg in get_args(annotation):
        if arg is type(None) or arg is Ellipsis:
            continue
        is_frozen_map = origin is FrozenMap or (
            isinstance(origin, type) and issubclass(origin, FrozenMap)
        )
        # FrozenMap's value parameter may be Any: it is an immutable holder.
        if is_frozen_map and arg is Any:
            continue
        problems.extend(_violations(arg, model_name, field_name))
    return problems


def test_no_mutable_container_annotation_at_any_depth():
    problems: list[str] = []
    hints_cache: dict[type[BaseModel], dict[str, Any]] = {}
    for model in sorted(_models(), key=lambda m: m.__name__):
        try:
            hints = hints_cache.setdefault(model, typing.get_type_hints(model))
        except NameError:  # pragma: no cover - unresolved forward refs
            hints = {}
        for field_name in model.model_fields:
            annotation = hints.get(field_name)
            if annotation is None:
                continue
            problems.extend(_violations(annotation, model.__name__, field_name))
    assert not problems, "mutable/opaque state fields:\n" + "\n".join(sorted(problems))


def test_every_state_model_is_frozen_and_strict():
    for model in sorted(_models(), key=lambda m: m.__name__):
        if not issubclass(model, Frozen):
            continue
        assert model.model_config.get("frozen") is True, model.__name__
        assert model.model_config.get("extra") == "forbid", model.__name__


def test_assignment_is_rejected(world):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        world.clock = None  # type: ignore[misc]


def test_nested_mapping_cannot_be_mutated_in_place(world):
    charge_id = next(iter(world.billing.charges))
    with pytest.raises(TypeError):
        world.billing.charges[charge_id] = None  # type: ignore[index]


def test_functional_update_does_not_alias_the_original(world):
    before = world.state_hash()
    advanced = world.advanced(5).stepped()
    assert world.state_hash() == before
    assert advanced.state_hash() != before


def test_sequences_are_tuples_and_sets_are_frozensets(world):
    ticket = next(iter(world.tickets.tickets.values()))
    assert isinstance(ticket.comments, tuple)
    assert isinstance(ticket.tags, (frozenset, AbstractSet))
    assert isinstance(world.billing.customers, Mapping)
    assert not isinstance(world.billing.customers, dict)
    assert isinstance(world.trace.entries, Sequence)


# ---------------------------------------------------------------------------
# validated updates - model_copy(update=...) bypasses validation entirely
# ---------------------------------------------------------------------------


def test_state_tools_and_env_never_use_unvalidated_model_copy():
    """``model_copy(update=...)`` installs whatever it is given, unchecked.

    That is an unacceptable primitive for state whose integrity the entire
    benchmark rests on, so the safety-critical packages must go through
    ``evolve`` instead.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "cerl"
    offenders = []
    for package in ("state", "tools", "env"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "model_copy"):
                    continue
                if not any(kw.arg == "update" for kw in node.keywords):
                    continue
                # ``super().model_copy(...)`` inside the WorldState override is
                # the one legitimate call: validation already happened upstream
                # in ``evolve``, and the override exists to reset memoised state.
                receiver = func.value
                if (
                    isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Name)
                    and receiver.func.id == "super"
                ):
                    continue
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, (
        "unvalidated updates in safety-critical packages:\n" + "\n".join(offenders)
    )


def test_model_copy_really_does_bypass_validation():
    """The premise of the rule above, asserted rather than assumed."""
    from cerl.state import Money

    corrupted = Money(cents=100).model_copy(update={"cents": "not a number"})
    assert corrupted.cents == "not a number"


def test_evolve_rejects_a_raw_dict_by_coercing_it_to_the_declared_type(world):
    """A raw dict must not survive into state as a mutable container."""
    from cerl.core import FrozenMap, evolve

    raw = {str(k): v for k, v in world.billing.customers.items()}
    assert isinstance(raw, dict)
    updated = evolve(world.billing, customers=raw)
    assert isinstance(updated.customers, FrozenMap)
    assert not isinstance(updated.customers, dict)
    with pytest.raises(TypeError):
        updated.customers["x"] = None  # type: ignore[index]


def test_evolve_rejects_a_wrong_prefix_id(world):
    from pydantic import ValidationError

    from cerl.core import evolve

    ticket = next(iter(world.tickets.tickets.values()))
    with pytest.raises(ValidationError):
        evolve(ticket, customer_id="tkt_000000000001")
    with pytest.raises(ValidationError):
        evolve(ticket, id="cus_000000000001")


def test_evolve_rejects_a_wrong_prefix_id_inside_a_nested_container(world):
    """A swapped key deep inside a mapping must not slip through."""
    from pydantic import ValidationError

    from cerl.core import evolve

    charges = {"tkt_000000000001": next(iter(world.billing.charges.values()))}
    with pytest.raises(ValidationError):
        evolve(world.billing, charges=charges)


def test_evolve_coerces_mutable_nested_containers(world):
    from cerl.core import FrozenMap, evolve

    customer = next(iter(world.billing.customers.values()))
    updated = evolve(customer, metadata={"tier": "gold"})
    assert isinstance(updated.metadata, FrozenMap)
    with pytest.raises(TypeError):
        updated.metadata["tier"] = "silver"  # type: ignore[index]

    ticket = next(iter(world.tickets.tickets.values()))
    with_tags = evolve(ticket, tags=["a", "b"])
    assert isinstance(with_tags.tags, frozenset)
    with_comments = evolve(ticket, comments=[])
    assert isinstance(with_comments.comments, tuple)


def test_evolve_rejects_invalid_model_values(world):
    from pydantic import ValidationError

    from cerl.core import evolve

    ticket = next(iter(world.tickets.tickets.values()))
    with pytest.raises(ValidationError):
        evolve(ticket, status="not_a_status")

    charge = next(iter(world.billing.charges.values()))
    with pytest.raises(ValidationError):
        evolve(charge, amount={"cents": "many"})
    with pytest.raises(ValidationError):
        evolve(charge, status="exploded")
    with pytest.raises(ValidationError):
        evolve(world, clock={"now": "soon"})


def test_evolve_rejects_an_undeclared_field(world):
    from cerl.core import UnknownField, evolve

    with pytest.raises(UnknownField):
        evolve(world, backdoor=1)


def test_evolve_accepts_a_well_formed_change_and_does_not_alias(world):
    from cerl.core import evolve

    before = world.state_hash()
    updated = evolve(world, clock=world.clock.advanced(3))
    assert world.state_hash() == before
    assert updated.clock.now == world.clock.now + 3


def test_every_mutating_tool_produces_a_validatable_world(all_frozen):
    """End-to-end: a full oracle episode leaves state that revalidates cleanly."""
    from cerl.reference import W2Oracle, run_reference
    from cerl.state import WorldState

    for scenario in all_frozen[:10]:
        final = run_reference(scenario, W2Oracle()).final
        revalidated = WorldState.model_validate(final.model_dump(mode="json"))
        assert revalidated.state_hash() == final.state_hash()


def test_memoised_derived_values_are_invisible(world):
    """Caching a hash on an immutable model must not change what it *is*.

    Private attributes are excluded from serialization, equality and hashing, so
    memoising the document and hash is safe here in a way it would not be on a
    mutable model.
    """
    dumped = world.model_dump(mode="json")
    assert "_document" not in dumped
    assert "_state_hash" not in dumped

    first = world.state_hash()
    world.as_document()
    assert world.state_hash() == first

    from cerl.core import content_hash

    document = dict(world.model_dump(mode="json"))
    document.pop("trace", None)
    assert first == content_hash(document), "the memoised hash must equal the computed one"


def test_a_copy_never_inherits_a_stale_derived_hash(world):
    """Regression: pydantic copies private attributes onto the new instance.

    Without an explicit reset, a mutated world reports the *previous* world's
    hash -- silently, and for every subsequent step.
    """
    from cerl.core import content_hash, evolve

    original = world.state_hash()
    world.as_document()  # populate the caches

    changed = evolve(world, clock=world.clock.advanced(7))
    assert changed.state_hash() != original

    document = dict(changed.model_dump(mode="json"))
    document.pop("trace", None)
    assert changed.state_hash() == content_hash(document)

    # And a plain model_copy is covered too, not just evolve.
    plain = world.model_copy(update={"clock": world.clock.advanced(9)})
    plain_doc = dict(plain.model_dump(mode="json"))
    plain_doc.pop("trace", None)
    assert plain.state_hash() == content_hash(plain_doc)
    assert plain.state_hash() != original
