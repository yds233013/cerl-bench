"""Pointer, matcher and diff-application semantics.

These are not coverage filler. Pointer matching is what decides whether a
business-diff op is permitted or becomes a residual, so a wrong answer here
moves a safety score in silence -- in either direction. Each test states the
decision the behaviour drives.
"""

from __future__ import annotations

import pytest

from cerl.diff import allowlist as allowlist_module
from cerl.diff.differ import apply_diff, diff_documents
from cerl.diff.models import DiffOp, DiffOpKind, Origin, StateDiff
from cerl.diff.pointer import (
    escape,
    get,
    get_field,
    join,
    matches,
    resolve_vars,
    split,
    unescape,
)
from cerl.diff.projection import BOOKKEEPING_PATHS, is_bookkeeping

# --------------------------------------------------------------------------
# pointer syntax
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pointer", ["", "/"])
def test_the_root_pointer_has_no_segments(pointer):
    """The whole-document pointer. ``apply_diff`` skips ops at the root rather
    than trying to replace the document out from under itself."""
    assert split(pointer) == ()


def test_a_pointer_must_be_absolute():
    """A relative pointer is a bug in the caller, not a pattern that matches
    nothing -- failing closed would hide it."""
    with pytest.raises(ValueError, match="must start with '/'"):
        split("billing/refunds")


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [("a/b", "a~1b"), ("a~b", "a~0b"), ("~1", "~01"), ("plain", "plain")],
)
def test_segments_containing_pointer_syntax_survive_a_round_trip(raw, escaped):
    """RFC 6901 escaping, and the order matters: unescaping ``~1`` before ``~0``
    would turn the literal text ``~01`` into ``/`` and silently retarget an op.
    """
    assert escape(raw) == escaped
    assert unescape(escaped) == raw
    assert split(join(raw)) == (raw,)


def test_integer_segments_are_accepted_and_stringified():
    assert join("comments", 0) == "/comments/0"


# --------------------------------------------------------------------------
# matching -- what the allowlist actually asks
# --------------------------------------------------------------------------


def test_a_pattern_matches_itself_and_everything_beneath_it():
    assert matches("/billing/refunds", "/billing/refunds")
    assert matches("/billing/refunds", "/billing/refunds/re_1/amount")


def test_a_pattern_does_not_match_a_shorter_pointer():
    """Permitting ``/billing/refunds/*/amount`` must not permit a write that
    replaces the whole refunds map."""
    assert not matches("/billing/refunds/*/amount", "/billing/refunds")


def test_a_wildcard_matches_one_segment_not_a_path():
    assert matches("/slack/channels/*/messages", "/slack/channels/support/messages")
    assert not matches("/slack/channels/*", "/slack/other/support")


def test_an_unresolvable_variable_fails_closed():
    """Documented behaviour: an unknown ``$.name`` is left literal, so it
    matches nothing and the op surfaces as a residual. An allowlist that
    silently widened on a typo is the failure that must not happen."""
    pattern = "/billing/charges/$.target_charge"
    assert resolve_vars(pattern, {}) == pattern
    assert not matches(pattern, "/billing/charges/ch_000000000001", {})
    assert matches(pattern, "/billing/charges/ch_000000000001",
                   {"target_charge": "ch_000000000001"})


# --------------------------------------------------------------------------
# reading values out of a document
# --------------------------------------------------------------------------


DOC = {
    "tickets": {"tkt_1": {"comments": [{"text": "first"}, {"text": "second"}]}},
    "count": 2,
}


def test_get_walks_arrays_by_index():
    assert get(DOC, "/tickets/tkt_1/comments/1/text") == "second"


@pytest.mark.parametrize(
    "pointer",
    [
        "/tickets/tkt_1/comments/9",       # index past the end
        "/tickets/tkt_1/comments/last",    # not an index
        "/tickets/missing",                # absent key
        "/count/cents",                    # descending into a scalar
    ],
)
def test_get_returns_none_rather_than_raising(pointer):
    """The allowlist reads constrained fields out of arbitrary ops. A miss must
    be a non-match, not an exception that aborts grading mid-episode."""
    assert get(DOC, pointer) is None


def test_get_field_reads_a_dotted_path_and_misses_quietly():
    assert get_field({"amount": {"cents": 500}}, "amount.cents") == 500
    assert get_field({"amount": {"cents": 500}}, "amount.currency") is None
    assert get_field({"amount": 500}, "amount.cents") is None


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", BOOKKEEPING_PATHS)
def test_every_bookkeeping_path_and_its_children_are_recognised(path):
    assert is_bookkeeping(path)
    assert is_bookkeeping(path + "/anything")


def test_a_business_path_that_merely_shares_a_prefix_is_not_bookkeeping():
    """``/clock`` is bookkeeping; a hypothetical ``/clockwork`` would not be.
    Prefix matching without the separator would exclude business state from
    grading entirely."""
    assert not is_bookkeeping("/clockwork")
    assert not is_bookkeeping("/billing/charges")


# --------------------------------------------------------------------------
# value constraints
# --------------------------------------------------------------------------


def test_a_value_constraint_accepts_a_set_of_permitted_values():
    match = allowlist_module._value_matches
    assert match("resolved", {"in": ["resolved", "escalated"]}, {})
    assert not match("open", {"in": ["resolved", "escalated"]}, {})


def test_money_is_compared_against_the_scenario_variable_in_cents():
    """Money is an object in the document and an integer in the variables; the
    constraint has to bridge that or every amount check would fail open."""
    match = allowlist_module._value_matches
    money = {"cents": 42000, "currency": "USD"}
    assert match(money, "$.duplicate_amount", {"duplicate_amount": 42000})
    assert not match(money, "$.duplicate_amount", {"duplicate_amount": 999})


# --------------------------------------------------------------------------
# StateDiff container and apply_diff
# --------------------------------------------------------------------------


def test_a_state_diff_behaves_like_the_sequence_it_is():
    op = DiffOp(op=DiffOpKind.ADD, path="/a", before=None, after=1,
                origin=Origin.AGENT)
    diff = StateDiff.of((op,))
    assert len(diff) == 1
    assert list(diff) == [op]
    assert diff
    assert not StateDiff.of(())


def test_apply_diff_round_trips_an_add_and_a_remove():
    before = {"billing": {"refunds": {}}, "tickets": {"t": {"status": "open"}}}
    after = {"billing": {"refunds": {"re_1": {"cents": 100}}}, "tickets": {}}
    assert apply_diff(before, diff_documents(before, after)) == after


def test_apply_diff_does_not_mutate_the_document_it_is_given():
    before = {"tickets": {"t": {"comments": [{"text": "a"}]}}}
    original = {"tickets": {"t": {"comments": [{"text": "a"}]}}}
    after = {"tickets": {"t": {"comments": [{"text": "a"}, {"text": "b"}]}}}
    apply_diff(before, diff_documents(before, after))
    assert before == original


def test_apply_diff_ignores_an_op_at_the_document_root():
    """A root op cannot be applied to a subtree of itself; skipping is the only
    coherent answer and it must not raise."""
    root = DiffOp(op=DiffOpKind.REPLACE, path="", before={"a": 1}, after={"b": 2},
                  origin=Origin.AGENT)
    assert apply_diff({"a": 1}, StateDiff.of((root,))) == {"a": 1}


def test_a_removed_key_that_was_never_there_is_not_an_error():
    remove = DiffOp(op=DiffOpKind.REMOVE, path="/missing", before=1, after=None,
                    origin=Origin.AGENT)
    assert apply_diff({"a": 1}, StateDiff.of((remove,))) == {"a": 1}
