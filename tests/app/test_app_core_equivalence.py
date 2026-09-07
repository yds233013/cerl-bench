"""The same actions through the app and through the core must agree.

If the workspace can diverge from the simulator, the simulator has stopped
being the single source of truth and nothing observed in the UI means anything.
"""

from __future__ import annotations

import pytest

from cerl.app import demos as demo_lib
from cerl.app.operational import Workspace
from cerl.reference.registry import oracle_for
from cerl.reference.runner import run_actions, run_reference
from cerl.verify import verify


@pytest.fixture(scope="module")
def w2(all_frozen):
    return [s for s in all_frozen if s.family == "duplicate_charge_approval"]


@pytest.fixture(scope="module")
def fixture():
    return demo_lib.load_demo_fixture() or demo_lib.build_demo_fixture()


@pytest.mark.parametrize(
    "demo_id",
    [
        "refund-with-valid-approval",
        "obtain-approval-then-refund",
        "escalate-without-authorization",
    ],
)
def test_the_app_reproduces_the_core_exactly(w2, fixture, demo_id):
    """Drive the oracle's actions through the app, then compare to the core."""
    workspace = Workspace(list(w2), fixture)
    demo = next(d for d in workspace.demos if d.demo_id == demo_id)
    scenario = {**{s.scenario_id: s for s in w2}, fixture.scenario_id: fixture}[
        demo.scenario_id
    ]

    core = run_reference(scenario, oracle_for(scenario))

    _, view = workspace._create_session({}, {"demo_id": demo_id})
    sid = view["session_id"]
    for index, action in enumerate(core.actions):
        workspace._act(
            {"session_id": sid},
            {
                "submission_id": f"s{index}",
                "action": action.model_dump(mode="json"),
            },
        )
    session = workspace.sessions[sid]
    replay = run_actions(scenario, core.actions)

    # Same logical time, same terminal state, same verdict.
    assert session.logical_time == int(replay.final.clock.now)
    assert session._env.world.state_hash() == replay.final.state_hash()
    assert session.declared_outcome == replay.verdict.declared_outcome

    through_app = verify(
        scenario,
        scenario.world,
        session._env.world,
        session._env.world.trace,
    )
    assert through_app.safe_completion == replay.verdict.safe_completion
    assert through_app.failure_class == replay.verdict.failure_class
    assert len(through_app.violations) == len(replay.verdict.violations)
    assert through_app.task_completion == replay.verdict.task_completion


def test_the_trace_built_through_the_app_is_a_valid_chain(w2, fixture):
    """Immutable, hash-chained traces are not weakened by the adapter."""
    workspace = Workspace(list(w2), fixture)
    _, view = workspace._create_session(
        {}, {"demo_id": "escalate-without-authorization"},
    )
    sid = view["session_id"]
    for index, action in enumerate(
        [
            {"kind": "tickets.get", "ticket_id": "tkt_000000000001"},
            {"kind": "billing.list_charges", "customer_id": "cus_000000000001"},
        ],
    ):
        workspace._act({"session_id": sid}, {"submission_id": f"a{index}", "action": action})
    trace = workspace.sessions[sid]._env.world.trace
    # Raises ChainBroken if an entry hash does not follow from its predecessor.
    trace.verify_chain()
    assert len(trace.agent_entries()) == 2


def test_a_demo_replays_identically_after_reset(w2, fixture):
    """Reset must restore the exact initial state, not merely a similar one."""
    workspace = Workspace(list(w2), fixture)
    _, view = workspace._create_session({}, {"demo_id": "refund-with-valid-approval"})
    sid = view["session_id"]
    action = {"kind": "tickets.get", "ticket_id": "tkt_000000000001"}

    workspace._act({"session_id": sid}, {"submission_id": "x", "action": action})
    first = workspace.sessions[sid]._env.world.state_hash()

    workspace._reset({"session_id": sid}, {})
    workspace._act({"session_id": sid}, {"submission_id": "y", "action": action})
    second = workspace.sessions[sid]._env.world.state_hash()

    assert first == second
