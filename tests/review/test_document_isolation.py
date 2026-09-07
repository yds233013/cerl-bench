"""Adversarial: no externally reachable document view may reach back into state.

The rule this file defends is one sentence: **nothing a caller can reach may
change a WorldState, a later document read, a state hash, or a trace seal.**

It is written as an attack rather than as a unit test because the previous two
failures here were both reachable through the ordinary public API and both
looked correct in review. The first sealed the agent's own tool result into the
trace; the second handed out the memoised document itself through a method typed
``Mapping``, which reads as read-only and is not -- its nested values are
ordinary ``dict`` and ``list``, so a nested write changed every later
``as_document()`` while ``state_hash()`` kept the original. A document and a hash
that disagree is the worst outcome available here, because both look fine.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import pytest

from cerl.actions import BillingIssueRefund, TicketsGet
from cerl.core import ChargeId, TicketId
from cerl.env.env import CerlEnv
from cerl.reference.registry import oracle_for
from cerl.reference.runner import run_reference
from cerl.state import WorldState

MARKER = "MUTATED-BY-AN-EXTERNAL-CALLER"


@pytest.fixture(scope="module")
def w2(all_frozen):
    return next(s for s in all_frozen if s.family == "duplicate_charge_approval")


class Attack(NamedTuple):
    """What the attack managed to do.

    ``written`` are edits the container accepted; ``refused`` are edits it
    rejected outright. Both are successes for the defence -- a sealed record
    refuses, a detached copy accepts and is simply not the record -- but
    ``written + refused == 0`` means the attack touched nothing and any
    assertion after it would pass vacuously.
    """

    written: int
    refused: int

    @property
    def attempts(self) -> int:
        return self.written + self.refused


def _poison(node: object) -> Attack:
    """Write MARKER into every string reachable from ``node``.

    Deliberately indiscriminate: the point is not to test one field but to walk
    everything a caller can reach and try to change all of it.
    """
    written = refused = 0
    stack = [node]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, dict):
            items = list(current.items())
        elif isinstance(current, list):
            items = list(enumerate(current))
        else:
            continue
        for key, value in items:
            if isinstance(value, str):
                try:
                    current[key] = MARKER  # type: ignore[index]
                    written += 1
                except TypeError:
                    refused += 1
            elif isinstance(value, (dict, list)):
                stack.append(value)
    return Attack(written, refused)


def _contains_marker(node: object) -> bool:
    stack = [node]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, str):
            if current == MARKER:
                return True
        elif isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


def _typed_state_fingerprint(world: WorldState) -> str:
    """Read the *typed* models, never a document, so this is independent."""
    parts = [
        f"{c.id}:{c.display_name}:{c.email}:{c.status.value}"
        for c in world.billing.customers.values()
    ]
    parts += [f"{t.id}:{t.status.value}:{len(t.comments)}" for t in world.tickets.tickets.values()]
    parts += [f"{c.id}:{c.amount.cents}:{c.status.value}" for c in world.billing.charges.values()]
    return "|".join(sorted(parts))


# --------------------------------------------------------------------------
# every externally reachable view of a document
# --------------------------------------------------------------------------


def test_no_public_accessor_hands_out_the_cache(w2):
    """The only public document accessor must be the copying one."""
    accessors = [
        name
        for name in dir(WorldState)
        if "document" in name and not name.startswith("_")
    ]
    assert accessors == ["as_document"]


def test_poisoning_as_document_changes_nothing(w2):
    world = w2.world
    typed_before = _typed_state_fingerprint(world)
    hash_before = world.state_hash()

    attack = _poison(world.as_document())
    assert attack.attempts > 0, "the attack touched nothing; it would pass vacuously"

    assert _typed_state_fingerprint(world) == typed_before
    assert world.state_hash() == hash_before
    assert not _contains_marker(world.as_document())


def test_poisoning_a_tool_result_payload_changes_nothing(w2):
    """The observation the agent is handed, and the trace entry it produced."""
    env = CerlEnv(w2)
    env.reset()
    result = env.step(TicketsGet(ticket_id=TicketId(str(w2.variables["ticket"]))))
    world = env.world
    typed_before = _typed_state_fingerprint(world)
    hash_before = world.state_hash()

    assert _poison(result.observation.result.payload.to_dict()).attempts > 0

    assert _typed_state_fingerprint(world) == typed_before
    assert world.state_hash() == hash_before
    assert not _contains_marker(world.trace.entries[0].result.payload.to_dict())
    world.trace.verify_chain()


def test_poisoning_a_recorded_diff_payload_changes_nothing(w2):
    """DiffOps live in the trace and hold document values.

    They used to hold the world's own objects, so an op was a writable handle
    into a frozen snapshot.
    """
    env = CerlEnv(w2)
    env.reset()
    charge = ChargeId(str(w2.variables["target_charge"]))
    env.step(
        BillingIssueRefund(
            charge_id=charge,
            amount_cents=int(w2.variables["duplicate_amount"]),
            reason="duplicate",
        ),
    )
    world = env.world
    typed_before = _typed_state_fingerprint(world)
    hash_before = world.state_hash()

    attempts = 0
    for entry in world.trace.entries:
        for op in entry.business_diff.ops:
            attempts += _poison(op.before).attempts + _poison(op.after).attempts
    assert attempts > 0, "no diff payload was reachable; the attack was vacuous"

    assert _typed_state_fingerprint(world) == typed_before
    assert world.state_hash() == hash_before
    assert not _contains_marker(world.as_document())
    world.trace.verify_chain()


def test_poisoning_an_exported_verdict_document_changes_nothing(w2):
    """Whatever a verdict or report hands back is a caller-reachable view too."""
    episode = run_reference(w2, oracle_for(w2))
    world = episode.final
    typed_before = _typed_state_fingerprint(world)
    hash_before = world.state_hash()

    attempts = _poison(episode.verdict.model_dump(mode="json")).attempts
    attempts += _poison(world.as_document()).attempts
    assert attempts > 0

    assert _typed_state_fingerprint(world) == typed_before
    assert world.state_hash() == hash_before
    assert not _contains_marker(world.as_document())
    episode.trace.verify_chain()


def test_the_hash_and_the_document_can_never_disagree(w2):
    """The precise failure the reviewed hole produced.

    state_hash() and as_document() are two views of one thing. If an external
    write can move one without the other, every downstream comparison is
    unsound, and it is unsound *quietly*.
    """
    world = w2.world
    for _ in range(3):
        _poison(world.as_document())

    from cerl.core import content_hash

    document = dict(world.as_document())
    document.pop("trace", None)
    assert content_hash(document) == world.state_hash()


# --------------------------------------------------------------------------
# and the isolation must not cost the time budget
# --------------------------------------------------------------------------


def test_the_isolation_does_not_break_the_episode_time_budget(all_frozen):
    """50 ms/episode, from docs/design.md. Not negotiable downward.

    Kept here as well as in tests/determinism because this file's isolation is
    the thing most likely to be paid for in wall time -- copying the world on
    every diff once took an episode to 77 ms.
    """
    import sys

    if sys.gettrace() is not None:
        pytest.skip("timing is not meaningful under a tracer (coverage/debugger)")

    scenario = max(all_frozen, key=lambda s: len(s.world.billing.charges))
    for _ in range(3):
        run_reference(scenario, oracle_for(scenario))

    batches = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(10):
            run_reference(scenario, oracle_for(scenario))
        batches.append((time.perf_counter() - start) / 10 * 1000)
    per_episode_ms = min(batches)
    assert per_episode_ms < 50, (
        f"{per_episode_ms:.1f} ms/episode exceeds the 50 ms budget "
        f"(batches: {[f'{b:.1f}' for b in batches]})"
    )
