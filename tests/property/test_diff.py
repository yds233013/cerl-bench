"""Layer 3 - diff: pointers, differ, projection, origin-aware allowlist."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from cerl.diff import (
    BOOKKEEPING_PATHS,
    DiffOp,
    DiffOpKind,
    Origin,
    PermittedDiff,
    apply_diff,
    diff_business,
    diff_documents,
    project_business,
    residual,
)
from cerl.diff.pointer import get_field, join, matches, resolve_vars, split

# --------------------------------------------------------------------------
# pointers
# --------------------------------------------------------------------------


def test_join_and_split_round_trip():
    assert split(join("billing", "refunds", "re_1")) == ("billing", "refunds", "re_1")


def test_escaping():
    assert split(join("a/b", "c~d")) == ("a/b", "c~d")


def test_wildcard_matches_one_segment():
    assert matches("/billing/refunds/*", "/billing/refunds/re_1")
    assert not matches("/billing/refunds/*", "/billing/charges/ch_1")


def test_pattern_matches_beneath_itself():
    # Subtree permits are how value-constrained allowlist entries work.
    assert matches("/billing/refunds/*", "/billing/refunds/re_1/amount/cents")


def test_pattern_does_not_match_shorter_pointer():
    assert not matches("/billing/refunds/*", "/billing/refunds")


def test_variable_substitution():
    assert resolve_vars("/tickets/$.ticket/status", {"ticket": "tkt_1"}) == "/tickets/tkt_1/status"


def test_unresolved_variable_fails_closed():
    # An allowlist that cannot resolve its variable must not accidentally permit.
    assert not matches("/tickets/$.ticket/status", "/tickets/tkt_1/status", {})


def test_get_field_dotted():
    assert get_field({"a": {"b": 1}}, "a.b") == 1
    assert get_field({"a": {"b": 1}}, "a.c") is None


# --------------------------------------------------------------------------
# differ
# --------------------------------------------------------------------------


def test_add_remove_replace():
    diff = diff_documents({"a": 1, "b": 2}, {"a": 9, "c": 3})
    ops = {(op.op, op.path) for op in diff.ops}
    assert (DiffOpKind.REPLACE, "/a") in ops
    assert (DiffOpKind.REMOVE, "/b") in ops
    assert (DiffOpKind.ADD, "/c") in ops


def test_diff_is_canonically_ordered():
    diff = diff_documents({}, {"z": 1, "a": 2, "m": 3})
    assert list(diff.paths()) == sorted(diff.paths())


def test_identical_documents_produce_empty_diff():
    doc = {"billing": {"charges": {"ch_1": {"amount": {"cents": 100}}}}}
    assert len(diff_documents(doc, doc)) == 0


def test_keyed_collection_insert_is_one_op():
    before = {"refunds": {}}
    after = {"refunds": {"re_1": {"amount": 5}}}
    diff = diff_documents(before, after)
    assert len(diff) == 1
    assert diff.ops[0].path == "/refunds/re_1"


@given(
    st.dictionaries(st.text(min_size=1, max_size=4), st.integers(), max_size=6),
    st.dictionaries(st.text(min_size=1, max_size=4), st.integers(), max_size=6),
)
def test_apply_diff_round_trip(before, after):
    assert apply_diff(before, diff_documents(before, after)) == after


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------


def test_projection_removes_exactly_the_bookkeeping_paths():
    document = {
        "meta": {"scenario_id": "s", "step_index": 4, "rng_counters": {"a": 1}},
        "clock": {"now": 10},
        "trace": {"entries": []},
        "violations": {"committed": [], "attempted": []},
        "tool_health": {"call_counts": {}},
        "responder_queue": {"pending": []},
        "billing": {"charges": {}},
        "slack": {},
        "tickets": {},
        "policy": {},
    }
    projected = project_business(document)
    assert set(projected) == {"meta", "billing", "slack", "tickets", "policy"}
    assert projected["meta"] == {"scenario_id": "s"}


def test_policy_is_business_state_not_bookkeeping():
    # A silent policy rewrite must be gradeable.
    assert "policy" in project_business({"policy": {"threshold": 1}})
    assert not any(p == "/policy" for p in BOOKKEEPING_PATHS)


def test_clock_churn_produces_no_business_diff():
    before = {"clock": {"now": 1}, "tool_health": {"call_counts": {"a": 1}}, "billing": {"x": 1}}
    after = {"clock": {"now": 99}, "tool_health": {"call_counts": {"a": 7}}, "billing": {"x": 1}}
    assert len(diff_business(before, after)) == 0


# --------------------------------------------------------------------------
# allowlist
# --------------------------------------------------------------------------


def _op(path: str, after=None, origin=Origin.AGENT, rule=None) -> DiffOp:
    return DiffOp(
        op=DiffOpKind.ADD, path=path, after=after, origin=origin, responder_rule=rule,
    )


def test_empty_allowlist_leaves_everything_residual():
    ops = [_op("/billing/refunds/re_1"), _op("/tickets/tkt_1/status")]
    assert len(residual(ops, [], {}).agent) == 2


def test_universal_allowlist_leaves_nothing_residual():
    ops = [_op("/billing/refunds/re_1"), _op("/tickets/tkt_1/status")]
    universal = [PermittedDiff(path="/*")]
    assert residual(ops, universal, {}).is_clean


def test_value_constraint_rejects_wrong_value():
    op = _op("/billing/refunds/re_1", after={"charge_id": "ch_other", "amount": {"cents": 42}})
    permitted = [
        PermittedDiff(
            path="/billing/refunds/*",
            constraints={"charge_id": "$.target_charge"},
        ),
    ]
    assert len(residual([op], permitted, {"target_charge": "ch_target"}).agent) == 1


def test_value_constraint_accepts_right_value():
    op = _op("/billing/refunds/re_1", after={"charge_id": "ch_target", "amount": {"cents": 42}})
    permitted = [
        PermittedDiff(
            path="/billing/refunds/*",
            constraints={"charge_id": "$.target_charge", "amount": "$.dup"},
        ),
    ]
    assert residual([op], permitted, {"target_charge": "ch_target", "dup": 42}).is_clean


def test_value_in_constraint():
    permitted = [PermittedDiff(path="/tickets/t/status", constraints={"value_in": ["resolved"]})]
    assert residual([_op("/tickets/t/status", after="resolved")], permitted, {}).is_clean
    assert len(residual([_op("/tickets/t/status", after="open")], permitted, {}).agent) == 1


# --- decision 3: no global responder exemption ---------------------------


def test_responder_op_is_not_exempt_by_default():
    op = _op("/slack/approvals/apr_1", origin=Origin.RESPONDER, rule="r_grants")
    agent_permit = [PermittedDiff(path="/slack/approvals/*")]  # origin defaults to agent
    split_result = residual([op], agent_permit, {})
    assert len(split_result.responder) == 1
    assert not split_result.agent  # never blamed on the agent


def test_responder_op_permitted_only_when_rule_declared():
    op = _op("/slack/approvals/apr_1", origin=Origin.RESPONDER, rule="r_grants")
    wrong_rule = [
        PermittedDiff(path="/slack/approvals/*", origin=Origin.RESPONDER, rule="r_denies"),
    ]
    right_rule = [
        PermittedDiff(path="/slack/approvals/*", origin=Origin.RESPONDER, rule="r_grants"),
    ]
    assert len(residual([op], wrong_rule, {}).responder) == 1
    assert residual([op], right_rule, {}).is_clean


def test_agent_op_cannot_be_permitted_by_a_responder_entry():
    # The laundering direction: an agent must not inherit a responder's permit.
    op = _op("/slack/approvals/apr_1", origin=Origin.AGENT)
    permitted = [
        PermittedDiff(path="/slack/approvals/*", origin=Origin.RESPONDER, rule="r_grants"),
    ]
    assert len(residual([op], permitted, {}).agent) == 1
