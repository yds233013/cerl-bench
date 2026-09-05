"""Package boundaries and the four privilege proofs.

Import contracts prove a module was not imported. They do not prove data did not
leak, so the privilege proofs below check the *rendered observation stream* and
the reachability of the gold corpus as well.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys

import pytest

from cerl.actions import Action
from cerl.agents.base import Agent, ScriptedAgent, is_unprivileged_agent
from cerl.core import PrivilegeViolation
from cerl.env import CerlEnv, render_observation
from cerl.env.observation import Observation
from cerl.reference import W2Oracle, run_reference
from cerl.reference.ground_truth import GroundTruthView, ReferencePolicy
from cerl.scenario.axes import AXIS_VALUES
from tests.helpers import REPO

SRC = REPO / "src" / "cerl"

BRANCH_NAMES = (
    "refund_now",
    "request_then_refund",
    "escalate_unapproved",
    "refund_below_threshold",
)
# Axis values distinctive enough that an appearance is real leakage rather than
# an ordinary English word colliding with a label.
DISTINCTIVE_AXIS_VALUES = (
    "missing_obtainable",
    "missing_unobtainable",
    "unauthorized_approver",
    "scope_exceeded",
    "refund_timeout_once",
    "search_flaky",
    "present_similar_name",
    "present_similar_email",
    "partial_refund_started",
    "ticket_already_commented",
)
RUBRIC_PREDICATES = (
    "refund_exists_for_charge",
    "refund_has_valid_approval_ref",
    "trace_ordering",
    "trace_contains",
    "retry_is_safe",
    "ticket_status_is",
    "message_posted_in",
    "no_refund_for_customer",
    "no_ops_on_entities",
    "no_prohibited_ops",
)
ALLOWLIST_LITERALS = ("/billing/refunds/*", "permitted_diffs", "$.target_charge")


# --------------------------------------------------------------------------
# import-linter contracts
# --------------------------------------------------------------------------


def test_import_linter_contracts_pass():
    proc = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint-imports"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _imports_of(package: str) -> set[str]:
    found: set[str] = set()
    for path in (SRC / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
    return found


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("agents", ("cerl.reference", "cerl.verify", "cerl.scenario")),
        ("verify", ("cerl.tools", "cerl.env", "cerl.agents", "cerl.reference")),
        ("env", ("cerl.reference",)),
        ("tools", ("cerl.reference",)),
        ("scenario", ("cerl.env", "cerl.tools", "cerl.verify", "cerl.agents", "cerl.reference")),
        (
            "core",
            ("cerl.state", "cerl.diff", "cerl.trace", "cerl.tools", "cerl.scenario",
             "cerl.verify", "cerl.env", "cerl.agents", "cerl.reference", "cerl.actions"),
        ),
    ],
)
def test_package_does_not_import(package: str, forbidden: tuple[str, ...]):
    imported = _imports_of(package)
    for banned in forbidden:
        assert not any(
            module == banned or module.startswith(banned + ".") for module in imported
        ), f"cerl.{package} imports {banned}"


def test_only_tools_mutate_world_state():
    """Contract 6, which import-linter cannot express."""
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        top = rel.parts[0]
        if top in {"tools", "state", "scenario", "reference"} or rel.name == "cli.py":
            continue
        text = path.read_text(encoding="utf-8")
        # env composes tool results and appends trace entries, which is its job;
        # what it must not do is edit business subsystems directly.
        for marker in ("world.billing.model_copy", "world.tickets.model_copy"):
            if marker in text:
                offenders.append(f"{rel}: {marker}")
    assert not offenders, "business-state mutation outside tools/:\n" + "\n".join(offenders)


def test_verifier_does_not_mutate_state(all_frozen):
    """Behavioural proof that verify is a pure read-only function."""
    from cerl.verify import verify

    for scenario in all_frozen[:15]:
        episode = run_reference(scenario, W2Oracle())
        before_initial = episode.initial.state_hash()
        before_final = episode.final.state_hash()
        before_trace = episode.trace.head_hash
        verify(scenario, episode.initial, episode.final, episode.trace)
        assert episode.initial.state_hash() == before_initial
        assert episode.final.state_hash() == before_final
        assert episode.trace.head_hash == before_trace


# --------------------------------------------------------------------------
# privilege proof 1 - the agent protocol is unprivileged
# --------------------------------------------------------------------------


def test_agent_protocol_is_unprivileged():
    signature = inspect.signature(Agent.act)
    parameters = [p for p in signature.parameters if p != "self"]
    assert parameters == ["observation"], parameters
    annotation = signature.parameters["observation"].annotation
    assert annotation in {Observation, "Observation"}


def test_reference_policy_has_a_different_arity_than_agent():
    """An oracle is not substitutable for an agent, nor an agent for an oracle."""
    agent_params = [p for p in inspect.signature(Agent.act).parameters if p != "self"]
    reference_params = [
        p for p in inspect.signature(ReferencePolicy.act).parameters if p != "self"
    ]
    assert agent_params == ["observation"]
    assert reference_params == ["observation", "truth"]
    assert agent_params != reference_params


def test_concrete_agents_expose_no_extra_injection_point():
    for cls in (ScriptedAgent,):
        parameters = [p for p in inspect.signature(cls.act).parameters if p != "self"]
        assert parameters == ["observation"], (cls.__name__, parameters)


def test_oracle_is_not_a_valid_agent():
    """The oracle must fail the unprivileged-agent check.

    Note ``isinstance`` against a runtime_checkable Protocol is *not* sufficient
    here: it checks method presence only, and the oracle does define ``act``.
    The arity is what separates them.
    """
    assert isinstance(W2Oracle(), Agent), "presence-only check is genuinely insufficient"
    assert not is_unprivileged_agent(W2Oracle())
    assert is_unprivileged_agent(ScriptedAgent(()))


# --------------------------------------------------------------------------
# privilege proof 2 - no ground truth in the observation stream
# --------------------------------------------------------------------------


def test_no_ground_truth_in_observations(all_frozen):
    """Checked over rendered strings, because the type system cannot see this.

    A payload field that happens to name the resolved branch is invisible to
    mypy and to import contracts; only reading what the agent would actually
    read catches it.
    """
    banned = (*BRANCH_NAMES, *DISTINCTIVE_AXIS_VALUES, *RUBRIC_PREDICATES, *ALLOWLIST_LITERALS)
    offenders: list[str] = []

    for scenario in all_frozen:
        env = CerlEnv(scenario)
        observation = env.reset()
        rendered = [render_observation(observation)]
        oracle = W2Oracle()
        from cerl.reference.ground_truth import ground_truth_for

        truth = ground_truth_for(scenario)
        while not env.done:
            action = oracle.act(observation, truth)
            observation = env.step(action).observation
            rendered.append(render_observation(observation))
        blob = "\n".join(rendered)
        offenders += [
            f"{scenario.scenario_id}: leaked {token!r}" for token in banned if token in blob
        ]

    assert not offenders, "ground truth reached the agent:\n" + "\n".join(offenders[:20])


def test_observation_has_no_branch_or_axis_fields():
    fields = set(Observation.model_fields)
    assert fields == {
        "brief", "result", "step_index", "steps_remaining", "logical_time", "notices",
    }
    for forbidden in ("branch", "axes", "facts", "rubric", "required_decision"):
        assert forbidden not in fields


def test_task_brief_says_the_workflow_not_the_variant(all_frozen):
    for scenario in all_frozen[:10]:
        brief = CerlEnv(scenario).reset().brief
        assert brief is not None
        assert scenario.branch not in brief.text
        for values in AXIS_VALUES.values():
            for value in values:
                if value in DISTINCTIVE_AXIS_VALUES:
                    assert value not in brief.text


# --------------------------------------------------------------------------
# privilege proof 3 - ground truth requires a token
# --------------------------------------------------------------------------


def test_ground_truth_requires_a_privileged_token(all_frozen):
    class Forged:
        pass

    with pytest.raises(PrivilegeViolation):
        GroundTruthView(all_frozen[0], Forged())  # type: ignore[arg-type]
    with pytest.raises(PrivilegeViolation):
        GroundTruthView(all_frozen[0], object())  # type: ignore[arg-type]


def test_token_type_is_not_exported():
    from cerl import reference

    assert not hasattr(reference, "_PrivilegedToken")
    assert "_PrivilegedToken" not in getattr(reference, "__all__", [])


# --------------------------------------------------------------------------
# privilege proof 4 - the gold corpus is unreachable from agents
# --------------------------------------------------------------------------


def _reachable_modules(start: str) -> set[str]:
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        module = frontier.pop()
        if module in seen or not module.startswith("cerl"):
            continue
        seen.add(module)
        parts = module.split(".")[1:]
        candidates = [SRC.joinpath(*parts).with_suffix(".py"), SRC.joinpath(*parts, "__init__.py")]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                frontier += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                frontier.append(node.module)
    return seen


def test_gold_corpus_not_reachable_from_agents():
    reachable: set[str] = set()
    for path in (SRC / "agents").rglob("*.py"):
        module = "cerl.agents." + path.stem if path.stem != "__init__" else "cerl.agents"
        reachable |= _reachable_modules(module)
    forbidden = {"cerl.reference.gold", "cerl.reference.ground_truth", "cerl.reference.mutations"}
    assert not (reachable & forbidden), reachable & forbidden
    assert not any(m.startswith("cerl.reference") for m in reachable)
    assert not any(m.startswith("cerl.verify") for m in reachable)


def test_agents_package_contains_no_privileged_helpers():
    text = "\n".join(p.read_text(encoding="utf-8") for p in (SRC / "agents").rglob("*.py"))
    for token in ("GroundTruthView", "ground_truth_for", "scenarios/gold", "FrozenScenario"):
        assert token not in text, token


def test_scripted_agent_only_sees_observations(all_frozen):
    scenario = all_frozen[0]
    seen: list[Action] = []

    class Recorder(ScriptedAgent):
        def act(self, observation: Observation) -> Action:
            assert isinstance(observation, Observation)
            seen.append(observation)
            return super().act(observation)

    from cerl.actions import Finish

    agent = Recorder((Finish(summary="done"),))
    env = CerlEnv(scenario)
    observation = env.reset()
    env.step(agent.act(observation))
    assert seen and all(isinstance(o, Observation) for o in seen)
