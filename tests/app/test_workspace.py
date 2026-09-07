"""The operational workspace: simulator contract and observation boundary.

No HTTP server is started. The routes are exercised as functions, so the suite
does not depend on a port being free or a process being up.
"""

from __future__ import annotations

import json

import pytest

from cerl.app import demos as demo_lib
from cerl.app.http import ApiError
from cerl.app.operational import FORBIDDEN_KEYS, Workspace
from cerl.core import TicketId
from cerl.eval import splits


@pytest.fixture(scope="module")
def w2(all_frozen):
    return [s for s in all_frozen if s.family == "duplicate_charge_approval"]


@pytest.fixture(scope="module")
def fixture():
    return demo_lib.load_demo_fixture() or demo_lib.build_demo_fixture()


@pytest.fixture
def workspace(w2, fixture):
    return Workspace(list(w2), fixture)


def _session(workspace: Workspace, demo: str = "escalate-without-authorization"):
    _, view = workspace._create_session({}, {"demo_id": demo})
    return view["session_id"], view


def _act(workspace: Workspace, sid: str, action: dict, submission: str = "sub-1"):
    return workspace._act(
        {"session_id": sid}, {"submission_id": submission, "action": action},
    )


# --------------------------------------------------------------------------
# the observation boundary
# --------------------------------------------------------------------------


def test_no_privileged_field_reaches_the_wire(workspace):
    """Checked on the serialised bytes, not the imports.

    An import test proves a module was not imported; it does not prove a value
    did not reach a payload. The risk here is the value.
    """
    sid, _view = _session(workspace)
    _act(workspace, sid, {"kind": "tickets.get", "ticket_id": "tkt_000000000001"})
    _, payload = workspace._get_session({"session_id": sid}, {})
    blob = json.dumps(payload)
    for key in FORBIDDEN_KEYS:
        assert f'"{key}"' not in blob, key


def test_the_workspace_starts_empty_rather_than_showing_the_world(workspace):
    """A support tool that displayed everything would hand over free information."""
    _, view = _session(workspace)
    observed = view["observed"]
    assert observed["tickets"] == []
    assert observed["charges"] == []
    assert observed["approvals"] == []


def test_records_appear_only_after_a_tool_returns_them(workspace):
    sid, _ = _session(workspace)
    _, before = workspace._get_session({"session_id": sid}, {})
    assert before["observed"]["tickets"] == []
    _act(workspace, sid, {"kind": "tickets.get", "ticket_id": "tkt_000000000001"})
    _, after = workspace._get_session({"session_id": sid}, {})
    assert len(after["observed"]["tickets"]) == 1


def test_the_operational_module_imports_nothing_privileged():
    import ast
    import inspect

    from cerl.app import operational, session

    for module in (operational, session):
        tree = ast.parse(inspect.getsource(module))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.add(getattr(node, "module", "") or "")
                names.update(a.name for a in node.names)
        for forbidden in ("cerl.reference", "cerl.verify", "Verdict"):
            assert not any(forbidden in n for n in names), (module.__name__, forbidden)


# --------------------------------------------------------------------------
# the simulator contract
# --------------------------------------------------------------------------


def test_reading_the_session_never_advances_logical_time(workspace):
    """Rendering, polling and waiting are passive. Only actions cost time."""
    sid, view = _session(workspace)
    start = view["logical_time"]
    for _ in range(10):
        _, polled = workspace._get_session({"session_id": sid}, {})
    assert polled["logical_time"] == start
    assert polled["step_index"] == 0


def test_an_action_advances_logical_time_by_its_documented_cost(workspace):
    from cerl.actions import ActionKind
    from cerl.tools import tick_cost

    sid, view = _session(workspace)
    start = view["logical_time"]
    _, res = _act(workspace, sid, {"kind": "tickets.get", "ticket_id": "tkt_000000000001"})
    assert res["session"]["logical_time"] == start + tick_cost(ActionKind.TICKETS_GET)


def test_one_submission_token_dispatches_exactly_once(workspace):
    """A double click, a re-render or a network retry must not act twice."""
    sid, _ = _session(workspace)
    action = {"kind": "tickets.get", "ticket_id": "tkt_000000000001"}
    _, first = _act(workspace, sid, action, submission="same")
    _, second = _act(workspace, sid, action, submission="same")
    assert first["session"]["step_index"] == second["session"]["step_index"] == 1
    assert first["record"]["index"] == second["record"]["index"]


def test_a_missing_submission_token_is_refused(workspace):
    sid, _ = _session(workspace)
    with pytest.raises(ApiError) as exc:
        workspace._act({"session_id": sid}, {"action": {"kind": "abstain", "reason": "x"}})
    assert exc.value.status == 400


def test_an_invalid_argument_is_reported_and_not_dispatched(workspace):
    """A person who mistypes gets an error, not a scored malformed action."""
    sid, _ = _session(workspace)
    with pytest.raises(ApiError) as exc:
        _act(workspace, sid, {
            "kind": "billing.issue_refund", "charge_id": "ch_000000000002",
            "amount_cents": 1, "reason": "not-a-real-reason",
        })
    assert exc.value.status == 422
    _, view = workspace._get_session({"session_id": sid}, {})
    assert view["step_index"] == 0


def test_reset_restores_the_exact_initial_state(workspace):
    sid, view = _session(workspace)
    start = view["logical_time"]
    _act(workspace, sid, {"kind": "tickets.get", "ticket_id": "tkt_000000000001"})
    _, reset = workspace._reset({"session_id": sid}, {})
    assert reset["logical_time"] == start
    assert reset["step_index"] == 0
    assert reset["history"] == []
    assert reset["observed"]["tickets"] == []


def test_sessions_are_isolated(workspace):
    a, _ = _session(workspace)
    b, _ = _session(workspace)
    _act(workspace, a, {"kind": "tickets.get", "ticket_id": "tkt_000000000001"})
    _, view_a = workspace._get_session({"session_id": a}, {})
    _, view_b = workspace._get_session({"session_id": b}, {})
    assert view_a["logical_time"] > view_b["logical_time"]
    assert view_b["observed"]["tickets"] == []


def test_a_genuine_backend_interlock_still_fires(workspace):
    """Real restrictions are kept. The app removes none of them."""
    sid, _ = _session(workspace)
    _, res = _act(workspace, sid, {
        "kind": "billing.issue_refund", "charge_id": "ch_000000000002",
        "amount_cents": 99_999_999, "reason": "duplicate",
    })
    assert res["record"]["outcome"] == "denied"
    assert res["record"]["denied_interlock"] == "over_refund"


def test_the_app_does_not_prevent_an_unauthorised_refund(workspace):
    """Invariant B1, at the application layer.

    Adding a UI or API check here would silently remove the dependent variable
    the benchmark exists to measure. An unauthorised refund must commit.
    """
    sid, _ = _session(workspace)
    _, res = _act(workspace, sid, {
        "kind": "billing.issue_refund", "charge_id": "ch_000000000002",
        "amount_cents": 42_000, "reason": "duplicate",
    })
    assert res["record"]["outcome"] == "committed"
    assert len(res["session"]["observed"]["refunds"]) == 1


def test_an_ended_episode_refuses_a_genuinely_new_action(workspace):
    sid, _ = _session(workspace)
    _act(workspace, sid, {"kind": "abstain", "reason": "done"}, submission="end")
    # A *well-formed* action, so this reaches the finished check. The id here
    # used to be "t", which is not a valid TicketId; that passed only because
    # the finished check ran before validation, and it would have kept passing
    # if the refusal had stopped working.
    with pytest.raises(ApiError) as exc:
        _act(
            workspace,
            sid,
            {"kind": "tickets.get", "ticket_id": str(TicketId.mint(1))},
            submission="after",
        )
    assert exc.value.status == 409


def test_retrying_the_terminal_action_returns_the_original_record(workspace):
    """A retry is not a new action, so the finished check must not see it first.

    The order used to be reversed: ``done`` was checked before the submission
    token, so the client that never received the response to the action that
    ended the episode got a 409 when it retried -- for an action the server had
    already executed. Resolving the token first makes the retry idempotent
    without weakening the refusal above.
    """
    sid, _ = _session(workspace)
    action = {"kind": "abstain", "reason": "done"}
    first_status, first = _act(workspace, sid, action, submission="end")
    retry_status, retry = _act(workspace, sid, action, submission="end")
    assert first_status == retry_status == 200
    assert first["record"]["index"] == retry["record"]["index"]


# --------------------------------------------------------------------------
# demos
# --------------------------------------------------------------------------


def test_three_demonstrations_are_offered(workspace):
    assert len(workspace.demos) == 3
    assert {d.demo_id for d in workspace.demos} == {
        "refund-with-valid-approval",
        "obtain-approval-then-refund",
        "escalate-without-authorization",
    }


def test_corpus_demos_come_from_the_training_partition(workspace, w2):
    by_id = {s.scenario_id: s for s in w2}
    for demo in workspace.demos:
        if demo.source != "corpus/train":
            continue
        assert splits.partition_of(by_id[demo.scenario_id]) is splits.Partition.TRAIN


def test_the_authored_fixture_is_labelled_development_exposed(workspace):
    """It must never read as held-out evaluation data."""
    demo = next(d for d in workspace.demos if d.source == "demo-fixture")
    assert "development-exposed" in demo.provenance
    assert "NOT held-out" in demo.provenance


def test_no_demo_uses_an_evaluation_or_validation_scenario(workspace, w2):
    """Held-out data is not spent on demonstrations."""
    by_id = {s.scenario_id: s for s in w2}
    for demo in workspace.demos:
        scenario = by_id.get(demo.scenario_id)
        if scenario is None:
            continue  # the authored fixture, which is outside the graded corpus
        assert splits.partition_of(scenario) is splits.Partition.TRAIN


def test_the_demo_fixture_lives_outside_the_graded_corpus(fixture):
    from cerl.scenario import corpus

    assert fixture.scenario_id.endswith("s90001")
    committed = {p.stem for p in corpus.CANONICAL.frozen.glob("*.json")}
    assert fixture.scenario_id not in committed


def test_an_unknown_demo_is_a_clear_error(workspace):
    with pytest.raises(ApiError) as exc:
        workspace._create_session({}, {"demo_id": "nope"})
    assert exc.value.status == 404


# --------------------------------------------------------------------------
# the single start command
# --------------------------------------------------------------------------


def test_the_static_site_refuses_paths_that_escape_its_root(tmp_path):
    """`..` in a URL must not read the repository, even on loopback."""
    from cerl.app.http import StaticSite

    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html>shell")
    secret = tmp_path / "outside.txt"
    secret.write_text("NOT FOR SERVING")

    site = StaticSite(root)
    body, content_type = site.resolve("/../outside.txt")
    assert b"NOT FOR SERVING" not in body
    assert content_type.startswith("text/html")


def test_an_unknown_path_falls_back_to_the_app_shell(tmp_path):
    """Client-side routes are not 404s."""
    from cerl.app.http import StaticSite

    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html>shell")
    body, _ = StaticSite(root).resolve("/some/client/route")
    assert b"shell" in body


def test_api_prefixes_are_never_answered_by_the_frontend():
    """Otherwise the operational server would return 200 and an HTML shell for
    `/review/episodes`, which reads as "the privileged API is here" when it is
    deliberately absent."""
    from cerl.app.http import API_PREFIXES

    assert "/review/" in API_PREFIXES
    assert "/api/" in API_PREFIXES


def test_serving_the_frontend_is_optional(tmp_path):
    from cerl.app.http import Router, serve

    server = serve(Router(), 0, "test", None)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
