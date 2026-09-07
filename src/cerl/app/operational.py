"""The operational API. Serves only what the environment interface exposes.

**This module must never be able to reach privileged material.** Not "must not
send it" -- must not be *able* to. It imports no verifier, no reference policy,
no ground truth, and it assembles its responses field by field rather than
dumping a scenario, so a field added to ``FrozenScenario`` later cannot leak
into a payload by default.

``tests/app`` asserts that over the actual JSON bytes, not over the imports:
a boundary that holds only by convention is not a boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from cerl.actions import Action
from cerl.app import demos as demo_lib
from cerl.app.http import ApiError, Router
from cerl.app.session import (
    EpisodeFinished,
    StaleGeneration,
    SubmissionConflict,
    WorkspaceSession,
)
from cerl.scenario.schema import FrozenScenario

_ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)

#: Fields that must never appear in an operational payload. Checked by test
#: against the serialised response, because the risk is a value reaching the
#: wire, not a name appearing in code.
FORBIDDEN_KEYS: tuple[str, ...] = (
    "branch",
    "required_decision",
    "rubric",
    "permitted_diffs",
    "invariants",
    "violations",
    "attempted_violations",
    "verdict",
    "failure_class",
    "oracle_tool_calls",
    "facts",
    "axes",
)


class Workspace:
    """Sessions, demos, and the routes over them."""

    def __init__(
        self,
        scenarios: list[FrozenScenario],
        fixture: FrozenScenario | None = None,
    ) -> None:
        self._scenarios = {s.scenario_id: s for s in scenarios}
        if fixture is not None:
            self._scenarios[fixture.scenario_id] = fixture
        self.demos = demo_lib.catalogue(scenarios, fixture)
        self._by_demo = {d.demo_id: d for d in self.demos}
        self.sessions: dict[str, WorkspaceSession] = {}
        #: Counter-based session ids, not random ones. Rule 1 bans ambient
        #: randomness from ``src/cerl/``, and a session handle has no need of
        #: it: a counter makes the workspace itself reproducible, which is
        #: useful when reconstructing what a person did.
        self._next_session = 0

    # -- routes -----------------------------------------------------------

    def router(self) -> Router:
        router = Router()
        router.add("GET", "/api/health", self._health)
        router.add("GET", "/api/demos", self._list_demos)
        router.add("POST", "/api/sessions", self._create_session)
        router.add("GET", "/api/sessions/{session_id}", self._get_session)
        router.add("POST", "/api/sessions/{session_id}/actions", self._act)
        router.add("POST", "/api/sessions/{session_id}/reset", self._reset)
        return router

    # -- handlers ---------------------------------------------------------

    def _health(self, _p: dict[str, str], _b: dict[str, Any]) -> tuple[int, Any]:
        return 200, {"status": "ok", "mode": "operational", "demos": len(self.demos)}

    def _list_demos(self, _p: dict[str, str], _b: dict[str, Any]) -> tuple[int, Any]:
        return 200, {"demos": [demo_lib.as_json(d) for d in self.demos]}

    def _create_session(
        self, _p: dict[str, str], body: dict[str, Any],
    ) -> tuple[int, Any]:
        demo_id = str(body.get("demo_id", ""))
        demo = self._by_demo.get(demo_id)
        if demo is None:
            raise ApiError(
                404, f"unknown demo {demo_id!r}",
                {"available": sorted(self._by_demo)},
            )
        scenario = self._scenarios.get(demo.scenario_id)
        if scenario is None:
            raise ApiError(500, f"demo {demo_id} names a scenario that is not loaded")
        self._next_session += 1
        session_id = f"{demo.demo_id}-{self._next_session:03d}"
        session = WorkspaceSession(session_id, scenario, demo_id)
        self.sessions[session_id] = session
        return 201, session.view()

    def _session(self, params: dict[str, str]) -> WorkspaceSession:
        session = self.sessions.get(params["session_id"])
        if session is None:
            raise ApiError(404, "no such session; start one from a demo")
        return session

    def _get_session(
        self, params: dict[str, str], _b: dict[str, Any],
    ) -> tuple[int, Any]:
        """A pure read. Rendering and polling must not advance logical time."""
        return 200, self._session(params).view()

    def _act(self, params: dict[str, str], body: dict[str, Any]) -> tuple[int, Any]:
        session = self._session(params)
        submission_id = str(body.get("submission_id") or "")
        if not submission_id:
            raise ApiError(
                400,
                "submission_id is required",
                "The client generates one token per user gesture so a retried "
                "request cannot dispatch the action twice.",
            )
        # The finished-episode check lives inside the session, *after* the
        # submission lookup. Checking it here rejected an honest retry of a
        # terminal action with 409, when the correct answer is the original
        # result: the retry is the same gesture.
        raw = body.get("action")
        if not isinstance(raw, dict) or "kind" not in raw:
            raise ApiError(400, "action must be an object with a 'kind'")
        try:
            action = _ACTION_ADAPTER.validate_python(raw)
        except ValidationError as error:
            # Invalid arguments are reported to the operator, not dispatched.
            # The environment would score them as a malformed action, which is
            # right for an agent and wrong for a person who mistyped a field.
            first = error.errors()[0]
            raise ApiError(
                422,
                "that action is not valid",
                {
                    "field": ".".join(str(p) for p in first.get("loc", ())),
                    "problem": first.get("msg", "invalid"),
                },
            ) from error

        # Optional: the episode generation the client composed against. Absent
        # means "whatever is current", which is right for a fresh gesture.
        raw_generation = body.get("generation")
        try:
            generation = None if raw_generation is None else int(raw_generation)
        except (TypeError, ValueError) as error:
            raise ApiError(400, "generation must be an integer") from error

        try:
            record, view = session.submit(action, submission_id, generation)
        except SubmissionConflict as error:
            raise ApiError(
                409, "that submission id was already used for a different action",
                str(error),
            ) from error
        except StaleGeneration as error:
            raise ApiError(
                409, "this session was reset while your request was in flight",
                str(error),
            ) from error
        except EpisodeFinished as error:
            raise ApiError(
                409, "this episode has ended", "Reset the demo to run it again.",
            ) from error
        return 200, {
            "record": {
                "index": record.index,
                "kind": record.kind,
                "outcome": record.outcome,
                "message": record.message,
                "logical_time": record.logical_time,
                "denied_interlock": record.denied_interlock,
            },
            "session": view,
        }

    def _reset(self, params: dict[str, str], _b: dict[str, Any]) -> tuple[int, Any]:
        session = self._session(params)
        session.reset()
        return 200, session.view()
