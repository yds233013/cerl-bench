"""The unprivileged prompt-only baseline.

No live model evaluation runs here. No spending budget is authorised in the
project or session, and holding an API key is not the same as being authorised
to spend, so every fixture below is **synthetic and labelled as such**. Nothing
in this file may be reported as a model result.
"""

from __future__ import annotations

import os

import pytest

from cerl.agents import (
    AnthropicClient,
    LiveEvaluationNotAuthorized,
    ModelResponse,
    PromptOnlyAgent,
    ScriptedClient,
    TranscriptCacheClient,
    TranscriptCacheMiss,
    all_tool_schemas,
    is_unprivileged_agent,
    live_evaluation_authorized,
    transcript_entries,
)
from cerl.agents.model_client import AUTHORIZATION_ENV, BUDGET_ENV
from cerl.core import FrozenMap
from cerl.env import CerlEnv
from cerl.eval import run_agent
from cerl.reference import oracle_for, run_actions, run_reference


def _oracle_as_scripted(scenario):
    """Turn a known-good trajectory into synthetic model turns.

    A fixture, not a model. It exists to exercise the integration end to end --
    tool-schema round trip, parsing, bounded stepping, transcript capture --
    without spending anything.
    """
    gold = run_reference(scenario, oracle_for(scenario))
    responses = []
    for action in gold.actions:
        payload = action.model_dump(mode="json")
        kind = str(payload.pop("kind"))
        responses.append(
            ModelResponse(
                text="",
                tool_name=kind.replace(".", "__"),
                tool_input=FrozenMap(payload),
                stop_reason="tool_use",
                source="synthetic",
            ),
        )
    return gold, responses


# --------------------------------------------------------------------------
# the agent is unprivileged
# --------------------------------------------------------------------------


def test_the_baseline_agent_is_unprivileged():
    agent = PromptOnlyAgent(ScriptedClient([]))
    assert is_unprivileged_agent(agent)


def test_the_agent_package_cannot_reach_the_verifier_even_indirectly():
    """The reach the import contract exists to prevent.

    Constructing an environment gives its importer a transitive path to
    ``verify`` and ``scenario``, so the policy package must not import one.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "cerl"
    reachable: set[str] = set()
    frontier = ["cerl.agents"]
    while frontier:
        module = frontier.pop()
        if module in reachable or not module.startswith("cerl"):
            continue
        reachable.add(module)
        parts = module.split(".")[1:]
        candidates = [
            root.joinpath(*parts).with_suffix(".py"),
            root.joinpath(*parts, "__init__.py"),
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                frontier += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                frontier.append(node.module)

    for forbidden in ("cerl.verify", "cerl.reference", "cerl.scenario"):
        offenders = {m for m in reachable if m == forbidden or m.startswith(forbidden + ".")}
        assert not offenders, f"agents can reach {forbidden}: {sorted(offenders)[:3]}"


def test_the_agent_sees_only_public_material(all_frozen):
    """What actually reaches the model: brief, observations, tool schemas."""
    scenario = all_frozen[0]
    _, responses = _oracle_as_scripted(scenario)
    client = ScriptedClient(responses)
    agent = PromptOnlyAgent(client, scenario_id=scenario.scenario_id)
    run_agent(CerlEnv(scenario), agent)

    # Distinctive labels only. Bare words like "expired" are also ordinary
    # English and appear legitimately in the *public* policy text ("has not
    # expired"); banning those would flag the policy the agent is meant to read.
    distinctive_axis_values = {
        "missing_obtainable", "missing_unobtainable", "missing_unanswered",
        "unauthorized_approver", "scope_exceeded", "refund_timeout_once",
        "search_flaky", "present_similar_name", "present_similar_email",
        "conflicting_external_ref", "name_only_similarity", "strong_match",
        "open_dispute_on_duplicate", "ticket_already_commented",
        "partial_refund_started", "above_threshold", "below_threshold",
    }
    banned = [
        scenario.branch,
        *[v for v in scenario.axes.values() if v in distinctive_axis_values],
        *[item.predicate for item in scenario.rubric],
        *[spec.path for spec in scenario.permitted_diffs],
    ]
    blob = "\n".join(
        str(request["system"]) + "\n" + "\n".join(str(m) for m in request["messages"])
        for request in client.requests
    )
    for token in banned:
        assert token not in blob, f"privileged material reached the model: {token!r}"


def test_tool_schemas_are_public_and_complete():
    schemas = all_tool_schemas()
    # 23 tools + finish/escalate/abstain; the malformed sentinel is not offered.
    assert len(schemas) == 26
    names = {s["name"] for s in schemas}
    assert "malformed" not in names
    blob = str(schemas)
    for token in ("branch", "rubric", "oracle", "verdict", "expected"):
        assert token not in blob.lower(), token


# --------------------------------------------------------------------------
# integration, on clearly labelled synthetic fixtures
# --------------------------------------------------------------------------


def test_the_agent_drives_a_full_episode_from_synthetic_turns(all_frozen):
    scenario = all_frozen[0]
    gold, responses = _oracle_as_scripted(scenario)
    agent = PromptOnlyAgent(ScriptedClient(responses), scenario_id=scenario.scenario_id)
    run = run_agent(CerlEnv(scenario), agent)

    assert run.steps == len(gold.actions)
    assert not run.stopped_early
    assert run.stop_reason == "declared"
    # The tool-schema round trip is lossless: the actions come back identical.
    assert [a.model_dump(mode="json") for a in run.actions] == [
        a.model_dump(mode="json") for a in gold.actions
    ]
    # And scoring them reproduces the oracle's verdict.
    episode = run_actions(scenario, run.actions)
    assert episode.verdict.is_clean_oracle_run


def test_every_transcript_is_labelled_synthetic(all_frozen):
    """A fixture must never be mistakable for a model result."""
    scenario = all_frozen[0]
    _, responses = _oracle_as_scripted(scenario)
    agent = PromptOnlyAgent(ScriptedClient(responses), scenario_id=scenario.scenario_id)
    run_agent(CerlEnv(scenario), agent)
    assert agent.transcript.source == "synthetic"
    assert all(e.response.source == "synthetic" for e in agent.transcript.entries)


def test_an_unparseable_turn_becomes_a_scored_malformed_action(all_frozen):
    """Not an exception: an unparseable turn is a real, measurable behaviour."""
    scenario = all_frozen[0]
    client = ScriptedClient(
        [
            ModelResponse(text="I would look at the ticket.", stop_reason="end_turn"),
            ModelResponse(
                text="", tool_name="billing__issue_refund",
                tool_input=FrozenMap({"charge_id": "not-an-id"}), stop_reason="tool_use",
            ),
            ModelResponse(
                text="", tool_name="escalate",
                tool_input=FrozenMap({"reason": "unsure", "to": "x"}), stop_reason="tool_use",
            ),
        ],
    )
    agent = PromptOnlyAgent(client, scenario_id=scenario.scenario_id)
    run = run_agent(CerlEnv(scenario), agent)
    kinds = [str(a.kind) for a in run.actions]
    assert kinds[0] == "malformed"
    assert kinds[1] == "malformed"  # wrong-prefix id rejected by the action schema
    assert kinds[2] == "escalate"


def test_episode_length_is_bounded(all_frozen):
    """A model that never declares an outcome must still terminate."""
    scenario = all_frozen[0]
    never_stops = ScriptedClient(
        [
            ModelResponse(
                text="", tool_name="tickets__get",
                tool_input=FrozenMap({"ticket_id": str(scenario.variables["ticket"])}),
                stop_reason="tool_use",
            ),
        ]
        * 200,
    )
    agent = PromptOnlyAgent(never_stops, scenario_id=scenario.scenario_id)
    run = run_agent(CerlEnv(scenario), agent, max_steps=12)
    assert run.steps == 12
    assert run.stopped_early
    assert run.stop_reason == "max_steps"


# --------------------------------------------------------------------------
# transcript-cache reproducibility: a SEPARATE claim from offline replay
# --------------------------------------------------------------------------


def test_transcript_cache_reproduces_the_same_actions(all_frozen):
    """Replaying a model's recorded turns reproduces its actions.

    This is the *model* reproducibility claim. It is weaker than, and tested
    apart from, the environment's offline replay: one says "the same turns give
    the same actions", the other says "the same actions give the same scores".
    """
    scenario = all_frozen[0]
    _, responses = _oracle_as_scripted(scenario)
    first_agent = PromptOnlyAgent(ScriptedClient(responses), scenario_id=scenario.scenario_id)
    first = run_agent(CerlEnv(scenario), first_agent)

    cache = TranscriptCacheClient(transcript_entries(first_agent.transcript))
    replay_agent = PromptOnlyAgent(cache, scenario_id=scenario.scenario_id)
    second = run_agent(CerlEnv(scenario), replay_agent)

    assert cache.hits == first.steps
    assert [a.model_dump(mode="json") for a in second.actions] == [
        a.model_dump(mode="json") for a in first.actions
    ]


def test_a_cache_miss_fails_loudly_and_never_falls_back_to_a_live_call(all_frozen):
    scenario = all_frozen[0]
    empty = TranscriptCacheClient({})
    agent = PromptOnlyAgent(empty, scenario_id=scenario.scenario_id)
    with pytest.raises(TranscriptCacheMiss):
        run_agent(CerlEnv(scenario), agent)


def test_a_mismatched_cache_entry_is_a_miss_not_a_wrong_answer(all_frozen):
    """Keys are content-addressed over the whole request, so drift cannot alias."""
    scenario = all_frozen[0]
    _, responses = _oracle_as_scripted(scenario)
    agent = PromptOnlyAgent(ScriptedClient(responses), scenario_id=scenario.scenario_id)
    run_agent(CerlEnv(scenario), agent)

    entries = transcript_entries(agent.transcript)
    wrong_key = "f" * 64
    shifted = dict.fromkeys([wrong_key], next(iter(entries.values())))
    with pytest.raises(TranscriptCacheMiss):
        run_agent(CerlEnv(scenario), PromptOnlyAgent(TranscriptCacheClient(shifted)))


def test_the_two_reproducibility_claims_are_independent(all_frozen):
    """Offline replay holds even when no transcript cache exists at all."""
    scenario = all_frozen[0]
    _, responses = _oracle_as_scripted(scenario)
    agent = PromptOnlyAgent(ScriptedClient(responses), scenario_id=scenario.scenario_id)
    run = run_agent(CerlEnv(scenario), agent)

    # Environment replay: actions in, identical scores out. No model involved.
    first = run_actions(scenario, run.actions)
    second = run_actions(scenario, run.actions)
    assert first.final.state_hash() == second.final.state_hash()
    assert first.verdict.failure_class == second.verdict.failure_class


# --------------------------------------------------------------------------
# live evaluation is not authorised
# --------------------------------------------------------------------------


def test_live_evaluation_is_not_authorized_in_this_project():
    assert not live_evaluation_authorized()


def test_credentials_alone_do_not_authorize_spending(monkeypatch):
    """The distinction the instruction turns on, asserted directly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.delenv(AUTHORIZATION_ENV, raising=False)
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    assert not live_evaluation_authorized()
    with pytest.raises(LiveEvaluationNotAuthorized):
        AnthropicClient()


def test_both_signals_are_required(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "500")
    monkeypatch.delenv(AUTHORIZATION_ENV, raising=False)
    assert not live_evaluation_authorized()

    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.setenv(BUDGET_ENV, "0")
    assert not live_evaluation_authorized()

    monkeypatch.setenv(BUDGET_ENV, "500")
    assert live_evaluation_authorized()


def test_no_test_here_performs_a_live_call():
    """Nothing in this module may reach the network.

    Checked over the parsed module rather than its own text: a source scan for
    a call pattern matches the assertion that performs the scan.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AnthropicClient"
    ]
    # The only construction is inside a pytest.raises block asserting refusal.
    assert len(constructions) == 1
    assert os.environ.get(AUTHORIZATION_ENV) != "1"
    assert not live_evaluation_authorized()
