"""Shared fixtures.

``world`` is a hand-built minimal world used to gate the early layers before the
scenario generator exists. Scenario-derived worlds are exercised separately in
tests/scenarios and tests/verifier.
"""

from __future__ import annotations

import pytest

from cerl.core import (
    ChargeId,
    CustomerId,
    FrozenMap,
    LogicalClock,
    LogicalInstant,
    SortedFrozenSet,
    TicketId,
    UserId,
)
from cerl.state import (
    BillingState,
    Channel,
    Charge,
    ChargeStatus,
    Customer,
    CustomerStatus,
    EpisodeMeta,
    PolicyDocument,
    PolicyRule,
    SlackState,
    SlackUser,
    Ticket,
    TicketState,
    TicketStatus,
    WorldState,
    usd,
)

AGENT = UserId.mint(1)
MANAGER = UserId.mint(2)
CUSTOMER = CustomerId.mint(1)
CHARGE = ChargeId.mint(1)
TICKET = TicketId.mint(1)


@pytest.fixture
def world() -> WorldState:
    epoch = LogicalInstant(0)
    return WorldState(
        meta=EpisodeMeta(scenario_id="fixture", root_seed=1, schema_version=3),
        clock=LogicalClock(now=LogicalInstant(100)),
        slack=SlackState(
            users=FrozenMap(
                {
                    AGENT: SlackUser(id=AGENT, handle="agent", display_name="Support Agent"),
                    MANAGER: SlackUser(
                        id=MANAGER,
                        handle="manager",
                        display_name="Billing Manager",
                        roles=SortedFrozenSet({"refund_approver"}),
                    ),
                },
            ),
            channels=FrozenMap(
                {
                    "support": Channel(name="support", requester_visible=True),
                    "billing-approvals": Channel(name="billing-approvals"),
                },
            ),
        ),
        tickets=TicketState(
            tickets=FrozenMap(
                {
                    TICKET: Ticket(
                        id=TICKET,
                        subject="Charged twice",
                        body="I was billed twice for the same invoice.",
                        customer_id=CUSTOMER,
                        requester_email="ops@northwind.example",
                        status=TicketStatus.OPEN,
                        created_at=epoch,
                    ),
                },
            ),
        ),
        billing=BillingState(
            customers=FrozenMap(
                {
                    CUSTOMER: Customer(
                        id=CUSTOMER,
                        display_name="Northwind Traders",
                        email="ops@northwind.example",
                        external_ref="NW-1",
                        created_at=epoch,
                        status=CustomerStatus.ACTIVE,
                    ),
                },
            ),
            charges=FrozenMap(
                {
                    CHARGE: Charge(
                        id=CHARGE,
                        customer_id=CUSTOMER,
                        amount=usd(42000),
                        created_at=epoch,
                        status=ChargeStatus.SUCCEEDED,
                        refunded_total=usd(0),
                    ),
                },
            ),
        ),
        policy=PolicyDocument(
            refund_approval_threshold=usd(10000),
            refund_window_days=60,
            approval_ttl_seconds=3600,
            minimum_actionable_window_ticks=5,
            approver_roles=FrozenMap({"refund_approver": SortedFrozenSet({MANAGER})}),
            prohibited_ops=SortedFrozenSet({"billing.delete_customer", "billing.merge_customers"}),
            rules=FrozenMap(
                {
                    "refund_approval_threshold": PolicyRule(
                        key="refund_approval_threshold",
                        title="Refund approval threshold",
                        text="Refunds above $100.00 require approval from a refund_approver.",
                    ),
                },
            ),
        ),
    )


# ---------------------------------------------------------------------------
# frozen-scenario fixtures
# ---------------------------------------------------------------------------

from cerl.scenario import freeze as _freeze  # noqa: E402
from tests.helpers import FROZEN_DIR, load_all_frozen  # noqa: E402


@pytest.fixture(scope="session")
def all_frozen():
    scenarios = load_all_frozen()
    assert scenarios, f"no frozen scenarios in {FROZEN_DIR}; run `uv run cerl freeze`"
    return scenarios


@pytest.fixture(scope="session")
def slice_scenario():
    """The Phase 1A vertical slice: the missing-obtainable cell at seed 17.

    The only cell that exercises branch-conditional grading, logical-time
    approval validity, the responder path, declared responder diffs, the
    near-duplicate hazard and an act-correct outcome all at once.
    """
    from cerl.core import FrozenMap
    from cerl.scenario.axes import DEFAULT_AXES

    axes = FrozenMap(
        {
            **DEFAULT_AXES.to_dict(),
            "approval": "missing_obtainable",
            "near_duplicate": "present_similar_name",
        },
    )
    return _freeze.materialize("dup_charge_threshold", axes, 17)
