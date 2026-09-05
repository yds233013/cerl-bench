"""Stripe-like billing subsystem state."""

from __future__ import annotations

from enum import StrEnum

from cerl.core import (
    ChargeId,
    CustomerId,
    DisputeId,
    FrozenMap,
    InvoiceId,
    LogicalInstant,
    PaymentMethodId,
    RefundId,
    UserId,
)
from cerl.state.common import Frozen, Money


class CustomerStatus(StrEnum):
    ACTIVE = "active"
    DELINQUENT = "delinquent"
    CLOSED = "closed"


class ChargeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class DisputeStatus(StrEnum):
    OPEN = "open"
    WON = "won"
    LOST = "lost"


class RefundReason(StrEnum):
    DUPLICATE = "duplicate"
    REQUESTED_BY_CUSTOMER = "requested_by_customer"
    FRAUDULENT = "fraudulent"


class PaymentMethod(Frozen):
    id: PaymentMethodId
    customer_id: CustomerId
    brand: str
    last4: str


class Customer(Frozen):
    id: CustomerId
    display_name: str
    email: str
    # The field that conflicts across tools; the axis that separates a genuine
    # duplicate from a near-duplicate that must not be touched.
    external_ref: str | None
    created_at: LogicalInstant
    status: CustomerStatus
    merged_into: CustomerId | None = None
    metadata: FrozenMap[str, str] = FrozenMap()


class Charge(Frozen):
    id: ChargeId
    customer_id: CustomerId
    amount: Money
    created_at: LogicalInstant
    status: ChargeStatus
    idempotency_key: str | None = None
    invoice_id: InvoiceId | None = None
    payment_method_id: PaymentMethodId | None = None
    refunded_total: Money
    description: str = ""


class Refund(Frozen):
    id: RefundId
    charge_id: ChargeId
    amount: Money
    reason: RefundReason
    created_at: LogicalInstant
    issued_by: UserId
    # Links refund -> approval. The verifier reads this to decide whether the
    # refund was authorised, so it is state, not commentary.
    approval_ref: str | None = None
    idempotency_key: str | None = None


class Invoice(Frozen):
    id: InvoiceId
    customer_id: CustomerId
    amount: Money
    created_at: LogicalInstant


class Dispute(Frozen):
    id: DisputeId
    charge_id: ChargeId
    customer_id: CustomerId
    status: DisputeStatus
    opened_at: LogicalInstant


class BillingState(Frozen):
    customers: FrozenMap[CustomerId, Customer]
    charges: FrozenMap[ChargeId, Charge]
    refunds: FrozenMap[RefundId, Refund] = FrozenMap()
    invoices: FrozenMap[InvoiceId, Invoice] = FrozenMap()
    disputes: FrozenMap[DisputeId, Dispute] = FrozenMap()
    payment_methods: FrozenMap[PaymentMethodId, PaymentMethod] = FrozenMap()

    def refunds_for_charge(self, charge_id: ChargeId) -> tuple[Refund, ...]:
        return tuple(r for r in self.refunds.values() if r.charge_id == charge_id)

    def charges_for_customer(self, customer_id: CustomerId) -> tuple[Charge, ...]:
        return tuple(c for c in self.charges.values() if c.customer_id == customer_id)

    def open_disputes_for_customer(self, customer_id: CustomerId) -> tuple[Dispute, ...]:
        return tuple(
            d
            for d in self.disputes.values()
            if d.customer_id == customer_id and d.status is DisputeStatus.OPEN
        )
