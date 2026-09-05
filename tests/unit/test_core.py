"""Layer 1 - core: canonical encoding, ids, clock, frozen map, keyed rng."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cerl.core import (
    ChargeId,
    CustomerId,
    EntityId,
    FrozenMap,
    InvalidEntityId,
    KeyedRng,
    LogicalClock,
    LogicalInstant,
    NonCanonicalValue,
    TicketId,
    canonical_json,
    content_hash,
    derive_int,
)

# --------------------------------------------------------------------------
# canonical json
# --------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_omits_whitespace():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_is_insertion_order_invariant():
    left = canonical_json({"x": 1, "y": {"q": 2, "p": 3}})
    right = canonical_json({"y": {"p": 3, "q": 2}, "x": 1})
    assert left == right


def test_canonical_json_rejects_floats_at_any_depth():
    with pytest.raises(NonCanonicalValue):
        canonical_json({"amount": 1.5})
    with pytest.raises(NonCanonicalValue):
        canonical_json({"a": [{"b": [0.1]}]})


def test_canonical_json_rejects_non_string_keys():
    with pytest.raises(NonCanonicalValue):
        canonical_json({1: "a"})


@given(st.dictionaries(st.text(), st.integers(), max_size=8))
def test_canonical_json_is_byte_stable(payload):
    assert canonical_json(payload) == canonical_json(dict(reversed(list(payload.items()))))


def test_content_hash_is_stable_and_sensitive():
    assert content_hash({"a": 1}) == content_hash({"a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


# --------------------------------------------------------------------------
# entity ids
# --------------------------------------------------------------------------


def test_id_accepts_correct_prefix():
    assert CustomerId("cus_0000000000ff") == "cus_0000000000ff"


def test_id_rejects_wrong_prefix():
    with pytest.raises(InvalidEntityId):
        CustomerId("tkt_0000000000ff")


def test_id_rejects_wrong_format():
    with pytest.raises(InvalidEntityId):
        CustomerId("cus_xyz")


def test_abstract_base_cannot_be_constructed():
    with pytest.raises(InvalidEntityId):
        EntityId("cus_0000000000ff")


def test_mint_is_deterministic():
    assert ChargeId.mint(255) == "ch_0000000000ff"
    assert ChargeId.mint(255) == ChargeId.mint(255)


def test_ids_of_different_types_are_not_interchangeable_at_runtime():
    # str equality still holds (they are str subclasses) but construction of
    # the wrong nominal type from a foreign id must fail. That is the runtime
    # half of the protection; mypy covers the static half (tests/typing).
    ticket = TicketId.mint(1)
    with pytest.raises(InvalidEntityId):
        CustomerId(str(ticket))


# --------------------------------------------------------------------------
# frozen map
# --------------------------------------------------------------------------


def test_frozen_map_is_immutable():
    fm = FrozenMap({"a": 1})
    with pytest.raises(TypeError):
        fm["b"] = 2  # type: ignore[index]


def test_frozen_map_functional_updates_do_not_alias():
    original = FrozenMap({"a": 1})
    updated = original.set("b", 2)
    assert dict(original) == {"a": 1}
    assert dict(updated) == {"a": 1, "b": 2}


def test_frozen_map_is_hashable_and_canonically_ordered():
    assert hash(FrozenMap({"a": 1, "b": 2})) == hash(FrozenMap({"b": 2, "a": 1}))
    assert list(FrozenMap({"b": 1, "a": 2})) == ["a", "b"]


def test_frozen_map_remove():
    assert dict(FrozenMap({"a": 1, "b": 2}).remove("a")) == {"b": 2}


# --------------------------------------------------------------------------
# clock
# --------------------------------------------------------------------------


def test_clock_advances_immutably():
    clock = LogicalClock(now=LogicalInstant(10))
    later = clock.advanced(5)
    assert clock.now == 10
    assert later.now == 15


def test_clock_cannot_move_backwards():
    with pytest.raises(ValueError, match="backwards"):
        LogicalClock(now=LogicalInstant(1)).advanced(-1)


def test_instant_seconds_conversion():
    assert LogicalInstant.from_seconds(3600) == 60
    with pytest.raises(ValueError, match="whole number"):
        LogicalInstant.from_seconds(30)


# --------------------------------------------------------------------------
# keyed rng
# --------------------------------------------------------------------------


def test_derivation_is_reproducible_and_domain_separated():
    assert derive_int(1, "world_gen", 0) == derive_int(1, "world_gen", 0)
    assert derive_int(1, "world_gen", 0) != derive_int(1, "tool_health", 0)
    assert derive_int(1, "world_gen", 0) != derive_int(2, "world_gen", 0)


def test_keyed_rng_position_is_independent_of_draw_history():
    a = KeyedRng(7, "world_gen")
    a.below(100)
    a.below(100)
    b = KeyedRng(7, "world_gen")
    assert a.at(5) == b.at(5)


def test_keyed_rng_sub_domains_are_independent():
    rng = KeyedRng(7, "world_gen")
    assert rng.sub("names").at(0) != rng.sub("amounts").at(0)
