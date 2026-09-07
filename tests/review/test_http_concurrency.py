"""Defects 3 and 5, exercised over real HTTP against the production classes.

Sequential direct calls would have passed on the old code. These start a server
and post to it, because the defect was in the threading of the server itself.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import closing

import pytest

import cerl.env.env as env_module
from cerl.app import demos as demo_lib
from cerl.app import http as app_http
from cerl.app import operational, reviewer


def _free_port() -> int:
    import socket

    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(  # noqa: S310 - loopback test server
            self.base + path,
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def get(self, path: str) -> tuple[int, object]:
        """Status plus a parsed body when there is one; HTML and empty bodies included."""
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as response:  # noqa: S310
                return response.status, _decode(response.read())
        except urllib.error.HTTPError as error:
            return error.code, _decode(error.read())


def _decode(raw: bytes) -> object:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode("utf-8", "replace")


@pytest.fixture(scope="module")
def workspace_server(all_frozen):
    scenarios = [s for s in all_frozen if s.family == "duplicate_charge_approval"]
    fixture = demo_lib.load_demo_fixture() or demo_lib.build_demo_fixture()
    workspace = operational.Workspace(scenarios, fixture)
    port = _free_port()
    server = app_http.serve(workspace.router(), port, "test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield workspace, Client(f"http://127.0.0.1:{port}")
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _session(client: Client) -> str:
    status, view = client.post("/api/sessions", {"demo_id": "escalate-without-authorization"})
    assert status == 201
    return str(view["session_id"])


ACTION = {"kind": "tickets.get", "ticket_id": "tkt_000000000001"}


def test_concurrent_identical_submissions_execute_exactly_once(workspace_server, monkeypatch):
    """The reported reproduction: two posts, one token, both executed.

    A delay inside the environment step widens the window so the requests
    genuinely overlap; it changes no action semantics.
    """
    workspace, client = workspace_server
    session_id = _session(client)
    session = workspace.sessions[session_id]

    real = env_module.CerlEnv.step
    calls: list[int] = []

    def slow(self, action):
        calls.append(self.world.meta.step_index)
        time.sleep(0.3)
        return real(self, action)

    monkeypatch.setattr(env_module.CerlEnv, "step", slow)

    results: list[tuple[int, dict]] = []

    def submit() -> None:
        results.append(
            client.post(
                f"/api/sessions/{session_id}/actions",
                {"submission_id": "SAME", "action": ACTION},
            ),
        )

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [status for status, _ in results] == [200, 200]
    assert len({body["record"]["index"] for _, body in results}) == 1
    assert len(calls) == 1, "the environment was stepped more than once"
    assert len(session.history) == 1
    assert len(session._env.world.trace.agent_entries()) == 1
    assert session._env.world.meta.step_index == 1


def test_distinct_concurrent_submissions_produce_a_coherent_serial_history(
    workspace_server,
):
    """Different gestures must both land, in some order, with no lost update."""
    workspace, client = workspace_server
    session_id = _session(client)
    session = workspace.sessions[session_id]

    actions = [
        {"kind": "tickets.get", "ticket_id": "tkt_000000000001"},
        {"kind": "billing.list_charges", "customer_id": "cus_000000000001"},
        {"kind": "policy.get_rule", "rule_key": "refund_approval_threshold"},
    ]
    results: list[tuple[int, dict]] = []

    def submit(index: int) -> None:
        results.append(
            client.post(
                f"/api/sessions/{session_id}/actions",
                {"submission_id": f"tok-{index}", "action": actions[index]},
            ),
        )

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(len(actions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(status == 200 for status, _ in results)
    assert len(session.history) == len(actions)
    assert sorted(r.index for r in session.history) == [0, 1, 2]
    assert len(session._env.world.trace.agent_entries()) == len(actions)
    session._env.world.trace.verify_chain()


def test_retrying_a_terminal_action_is_idempotent(workspace_server):
    """It returned 409, because `done` was checked before the token lookup."""
    _, client = workspace_server
    session_id = _session(client)
    action = {"kind": "abstain", "reason": "done"}
    first = client.post(
        f"/api/sessions/{session_id}/actions", {"submission_id": "T", "action": action},
    )
    retry = client.post(
        f"/api/sessions/{session_id}/actions", {"submission_id": "T", "action": action},
    )
    assert first[0] == retry[0] == 200
    assert first[1]["record"]["index"] == retry[1]["record"]["index"]


def test_a_genuinely_new_action_after_the_end_is_still_refused(workspace_server):
    _, client = workspace_server
    session_id = _session(client)
    client.post(
        f"/api/sessions/{session_id}/actions",
        {"submission_id": "end", "action": {"kind": "abstain", "reason": "done"}},
    )
    status, body = client.post(
        f"/api/sessions/{session_id}/actions",
        {"submission_id": "after", "action": ACTION},
    )
    assert status == 409
    assert "ended" in body["error"]


def test_a_token_reused_with_a_different_action_is_refused(workspace_server):
    """It used to return the unrelated original record."""
    _, client = workspace_server
    session_id = _session(client)
    client.post(
        f"/api/sessions/{session_id}/actions", {"submission_id": "R", "action": ACTION},
    )
    status, body = client.post(
        f"/api/sessions/{session_id}/actions",
        {
            "submission_id": "R",
            "action": {"kind": "policy.get_rule", "rule_key": "refund_approval_threshold"},
        },
    )
    assert status == 409
    assert "different action" in body["error"]


def test_a_request_from_before_a_reset_cannot_act_on_the_new_episode(workspace_server):
    """Reset/action overlap must not mix episodes."""
    _, client = workspace_server
    session_id = _session(client)
    _, view = client.get(f"/api/sessions/{session_id}")
    generation = view["generation"]
    client.post(f"/api/sessions/{session_id}/reset", {})
    status, body = client.post(
        f"/api/sessions/{session_id}/actions",
        {"submission_id": "stale", "action": ACTION, "generation": generation},
    )
    assert status == 409
    assert "reset" in body["error"]


# --------------------------------------------------------------------------
# 5. reviewer routing in the packaged application
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reviewer_server(all_frozen):
    library = reviewer.ReviewLibrary(list(all_frozen))
    from tests.helpers import REPO

    dist = REPO / "app" / "dist"
    site = app_http.StaticSite(dist) if (dist / "index.html").exists() else None
    port = _free_port()
    server = app_http.serve(library.router(), port, "test-reviewer", site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield Client(f"http://127.0.0.1:{port}"), site
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_the_reviewer_serves_the_frontend_so_its_api_is_same_origin(reviewer_server):
    """The built bundle requests a relative /review/*.

    Serving only the API meant the packaged reviewer page could never reach it:
    opened from the operational port, its relative request hit a server that
    deliberately 404s those routes, and the cross-port routing existed only in
    Vite's development proxy.
    """
    client, site = reviewer_server
    if site is None:
        pytest.skip("frontend not built; run `cd app && npm run build`")
    assert client.get("/")[0] == 200
    assert client.get("/?review")[0] == 200
    status, body = client.get("/review/episodes")
    assert status == 200
    assert len(body["episodes"]) >= 1


def test_the_c_auth_example_is_reachable_from_the_reviewer_origin(reviewer_server):
    client, _ = reviewer_server
    status, body = client.get("/review/episodes/local-baseline:0")
    assert status == 200
    assert body["safety"]["failure_class"] == "UNAUTHORIZED_ACTION"
    assert [v["cost_class"] for v in body["safety"]["committed_violations"]] == ["C_AUTH"]


def test_the_operational_server_still_rejects_reviewer_routes(workspace_server):
    """Routing was not fixed by exposing privileged data on the operational API."""
    _, client = workspace_server
    assert client.get("/review/episodes")[0] == 404
    assert client.get("/review/health")[0] == 404


def test_a_malformed_generation_is_a_client_error_not_a_crash(workspace_server):
    _, client = workspace_server
    session_id = _session(client)
    status, body = client.post(
        f"/api/sessions/{session_id}/actions",
        {"submission_id": "bad-gen", "action": ACTION, "generation": "soon"},
    )
    assert status == 400
    assert "generation" in body["error"]
