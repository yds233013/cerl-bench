"""Static and runtime behaviour of validated nominal identifiers.

Checked under ``mypy --strict`` as well as at runtime. The static half is what a
future refactor back to ``NewType`` would break, so it is pinned here rather than
left as a comment.
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_type

import pytest

from cerl.actions import BillingGetCustomer, TicketsGet
from cerl.core import ChargeId, CustomerId, InvalidEntityId, TicketId


def test_ids_are_str_subclasses_with_their_own_identity() -> None:
    customer = CustomerId.mint(1)
    assert_type(customer, CustomerId)
    assert isinstance(customer, str)
    assert type(customer) is CustomerId


def test_a_bare_str_is_not_assignable_to_an_id() -> None:
    # mypy --strict rejects the next line; the ignore is the assertion.
    customer: CustomerId = "cus_000000000001"  # type: ignore[assignment]
    assert customer == "cus_000000000001"


def test_a_foreign_id_is_not_assignable() -> None:
    ticket = TicketId.mint(1)
    # A TicketId where a CustomerId is required is a static error. Both ignores
    # are the assertion: the assignment is rejected, and mypy further knows the
    # two nominal types can never compare equal.
    customer: CustomerId = ticket  # type: ignore[assignment]
    assert str(customer) == str(ticket)


def test_wrong_prefix_raises_at_construction() -> None:
    with pytest.raises(InvalidEntityId):
        CustomerId(str(TicketId.mint(1)))
    with pytest.raises(InvalidEntityId):
        ChargeId("cus_000000000001")


def test_wrong_prefix_raises_at_model_validation() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BillingGetCustomer.model_validate({"customer_id": "tkt_000000000001"})
    with pytest.raises(ValidationError):
        TicketsGet.model_validate({"ticket_id": "cus_000000000001"})


def test_wrong_prefix_raises_at_scenario_load(tmp_path: Path) -> None:
    """A swapped id in a frozen file fails at load, not in a verdict."""
    import json

    from pydantic import ValidationError
    from tests.helpers import frozen_paths

    from cerl.core import ScenarioDefect
    from cerl.scenario import freeze

    payload = json.loads(frozen_paths()[0].read_text(encoding="utf-8"))
    payload["variables"]["target_charge"] = "cus_000000000001"
    ticket_ids = list(payload["world"]["tickets"]["tickets"])
    payload["world"]["tickets"]["tickets"][ticket_ids[0]]["customer_id"] = "tkt_000000000001"
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((InvalidEntityId, ScenarioDefect, ValidationError)):
        freeze.load(corrupted)


def test_ids_serialize_as_plain_strings() -> None:
    action = BillingGetCustomer(customer_id=CustomerId.mint(3))
    dumped = action.model_dump(mode="json")
    assert dumped["customer_id"] == "cus_000000000003"
    assert isinstance(dumped["customer_id"], str)


def test_newtype_is_not_used_for_ids() -> None:
    """NewType is runtime-erased and would let a swapped id ride undetected.

    Parsed rather than grepped: the module docstring explains *why* NewType was
    rejected, so a text search would match its own rationale.
    """
    import ast
    import inspect

    from cerl.core import ids

    tree = ast.parse(inspect.getsource(ids))
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "NewType" not in names
    for id_type in (CustomerId, ChargeId, TicketId):
        assert issubclass(id_type, str)
        assert hasattr(id_type, "PATTERN")
