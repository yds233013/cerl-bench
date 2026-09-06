"""Foundation layer: identity, time, canonical encoding, keyed randomness.

``core`` imports nothing else from ``cerl`` (import-linter contract 5).
"""

from cerl.core.canonical import (
    canonical_bytes,
    canonical_json,
    content_hash,
    hash_text,
    report_hash,
)
from cerl.core.clock import SECONDS_PER_TICK, TICKS_PER_DAY, LogicalClock, LogicalInstant
from cerl.core.constraints import CONSTRAINT_ORDER, ConstraintClass
from cerl.core.errors import (
    CerlFault,
    ChainBroken,
    ExternalInterruption,
    FrozenMapMutation,
    InvalidEntityId,
    NonCanonicalValue,
    PrivilegeViolation,
    ReplayDivergence,
    ScenarioDefect,
)
from cerl.core.evolve import UnknownField, evolve, validate_field
from cerl.core.frozen_map import FrozenMap
from cerl.core.frozen_set import SortedFrozenSet
from cerl.core.ids import (
    ALL_ID_TYPES,
    ApprovalId,
    ChargeId,
    CustomerId,
    DisputeId,
    EntityId,
    InvoiceId,
    MessageId,
    PaymentMethodId,
    RefundId,
    TicketId,
    UserId,
)
from cerl.core.model import Frozen
from cerl.core.rng import (
    KeyedRng,
    derive_below,
    derive_bytes,
    derive_choice,
    derive_int,
    derive_range,
)
from cerl.core.vocabulary import RefundReason

__all__ = [
    "ALL_ID_TYPES",
    "CONSTRAINT_ORDER",
    "SECONDS_PER_TICK",
    "TICKS_PER_DAY",
    "ApprovalId",
    "CerlFault",
    "ChainBroken",
    "ChargeId",
    "ConstraintClass",
    "CustomerId",
    "DisputeId",
    "EntityId",
    "ExternalInterruption",
    "Frozen",
    "FrozenMap",
    "FrozenMapMutation",
    "InvalidEntityId",
    "InvoiceId",
    "KeyedRng",
    "LogicalClock",
    "LogicalInstant",
    "MessageId",
    "NonCanonicalValue",
    "PaymentMethodId",
    "PrivilegeViolation",
    "RefundId",
    "RefundReason",
    "ReplayDivergence",
    "ScenarioDefect",
    "SortedFrozenSet",
    "TicketId",
    "UnknownField",
    "UserId",
    "canonical_bytes",
    "canonical_json",
    "content_hash",
    "derive_below",
    "derive_bytes",
    "derive_choice",
    "derive_int",
    "derive_range",
    "evolve",
    "hash_text",
    "report_hash",
    "validate_field",
]
