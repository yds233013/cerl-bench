"""R1-R6: seeds, replay, regeneration, cross-process and cross-version agreement."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cerl.actions import BillingGetCharge, BillingIssueRefund, TicketsGet
from cerl.core import ChargeId, ReplayDivergence, TicketId, hash_text
from cerl.env import CerlEnv, replay
from cerl.reference import W2Oracle, oracle_for, run_reference
from cerl.reference.gold import load as load_gold
from cerl.scenario import freeze as freeze_module
from cerl.scenario.plan import all_instances
from cerl.state import FailureKind, ToolHealthState
from tests.helpers import GOLD_DIR, frozen_paths

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# R1 - counter-based, per-tool failure injection
# --------------------------------------------------------------------------


def test_failure_injection_is_independent_of_interleaved_calls(all_frozen):
    """The property that makes two policies comparable.

    With a shared RNG stream an agent making one extra read would shift every
    downstream failure, so two policies would face different worlds. Keyed on the
    per-tool call index, unrelated calls cannot move another tool's schedule.
    """
    scenario = next(
        s for s in all_frozen if s.axes.get("tool_reliability") == "refund_timeout_once"
    )
    charge = ChargeId(str(scenario.variables["target_charge"]))
    amount = int(scenario.variables["duplicate_amount"])
    original = ChargeId(str(scenario.variables["original_charge"]))

    direct = CerlEnv(scenario)
    direct.reset()
    straight = direct.step(BillingIssueRefund(charge_id=charge, amount_cents=amount))

    padded = CerlEnv(scenario)
    padded.reset()
    for _ in range(5):
        padded.step(BillingGetCharge(charge_id=original))
    after_padding = padded.step(BillingIssueRefund(charge_id=charge, amount_cents=amount))

    assert straight.observation.result.outcome == after_padding.observation.result.outcome


def test_permanently_failing_tool_never_succeeds_on_retry(all_frozen):
    scenario = next(
        s for s in all_frozen if s.axes.get("tool_reliability") == "search_flaky"
    )
    # Built through validation, not model_copy: model_copy(update=...) bypasses
    # validators, so a raw dict would reach FrozenMap's serializer untouched.
    health = ToolHealthState.model_validate(
        {
            "schedule": {
                "billing.get_charge": {str(i): FailureKind.UNAVAILABLE for i in range(10)},
            },
            "call_counts": {},
        },
    )
    world = scenario.world.model_copy(update={"tool_health": health})
    broken = scenario.model_copy(update={"world": world})

    env = CerlEnv(broken)
    env.reset()
    charge = ChargeId(str(scenario.variables["original_charge"]))
    for _ in range(6):
        result = env.step(BillingGetCharge(charge_id=charge))
        assert not result.observation.result.ok


# --------------------------------------------------------------------------
# R5/R6 - same seed, same terminal hash; across processes
# --------------------------------------------------------------------------


def test_oracle_is_byte_identical_within_a_process(all_frozen):
    for scenario in all_frozen[:12]:
        a = run_reference(scenario, oracle_for(scenario))
        b = run_reference(scenario, oracle_for(scenario))
        assert a.final.state_hash() == b.final.state_hash()
        assert a.trace.head_hash == b.trace.head_hash


def test_oracle_is_byte_identical_across_processes():
    """Run in a separate interpreter with a different hash seed.

    Set-iteration order is salted per process, so this is the check that catches
    an unordered container leaking into serialized state -- a bug that is
    completely invisible to a same-process comparison.
    """
    script = (
        "from cerl.scenario import freeze;"
        "from cerl.reference import W2Oracle, oracle_for, run_reference;"
        "from pathlib import Path;"
        "import json;"
        "ps = sorted(Path('scenarios/frozen').glob('*.json'))[:12];"
        "out = [];"
        "[out.append(run_reference(s, oracle_for(s)).final.state_hash()) "
        "for s in (freeze.load(p) for p in ps)];"
        "print(json.dumps(out))"
    )
    runs = []
    for hash_seed in ("0", "1", "987654"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed,
                 "PYTHONPATH": str(REPO / "src")},
        )
        runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert runs[0] == runs[1] == runs[2]
    assert len(runs[0]) == 12


def test_state_hash_is_stable_across_hash_seeds():
    frozen = "".join(sorted(p.read_text() for p in frozen_paths()))
    script = (
        "from pathlib import Path;"
        "from cerl.core import hash_text;"
        "print(hash_text(''.join(sorted(p.read_text() "
        "for p in Path('scenarios/frozen').glob('*.json')))))"
    )
    digests = set()
    for hash_seed in ("0", "42"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed,
                 "PYTHONPATH": str(REPO / "src")},
        )
        digests.add(proc.stdout.strip())
    assert len(digests) == 1
    assert digests.pop() == hash_text(frozen)


# --------------------------------------------------------------------------
# R4 - frozen snapshots regenerate byte-exactly
# --------------------------------------------------------------------------


def test_every_frozen_scenario_regenerates_byte_exactly():
    """Guards the two-representation drift risk of mutators + frozen snapshots."""
    from cerl.reference import produce

    for axes, seed in all_instances():
        scenario = freeze_module.materialize("dup_charge_threshold", axes, seed)
        _, gold = produce(scenario)
        scenario = scenario.model_copy(update={"oracle_tool_calls": gold.tool_calls})
        path = Path("scenarios/frozen") / f"{scenario.scenario_id}.json"
        assert path.exists(), f"{scenario.scenario_id} is not committed"
        assert freeze_module.to_json(scenario) == path.read_text(encoding="utf-8")


def test_manifest_matches_files_on_disk():
    manifest = json.loads(Path("scenarios/manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(frozen_paths())
    families = set()
    for entry in manifest["scenarios"]:
        path = Path("scenarios/frozen") / entry["file"]
        assert hash_text(path.read_text(encoding="utf-8")) == entry["sha256"]
        families.add(entry["family"])
    assert families == set(manifest["families"])


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def test_replay_reproduces_every_entry_including_responders(all_frozen):
    for scenario in all_frozen:
        episode = run_reference(scenario, oracle_for(scenario))
        world = replay(scenario, episode.trace)
        assert world.state_hash() == episode.final.state_hash()
        assert [e.entry_hash for e in world.trace.entries] == [
            e.entry_hash for e in episode.trace.entries
        ]


def test_replay_reproduces_responder_timing_and_queue(slice_scenario):
    episode = run_reference(slice_scenario, W2Oracle())
    responder_entries = episode.trace.responder_entries()
    assert responder_entries, "the slice must actually fire a responder"

    world = replay(slice_scenario, episode.trace)
    replayed = world.trace.responder_entries()
    assert len(replayed) == len(responder_entries)
    for want, got in zip(responder_entries, replayed, strict=True):
        assert (want.idx, int(want.logical_time), want.responder_rule) == (
            got.idx, int(got.logical_time), got.responder_rule,
        )
    assert world.responder_queue == episode.final.responder_queue


def test_replay_detects_divergence(slice_scenario, all_frozen):
    """Replaying a trace against a different world must not silently succeed."""
    episode = run_reference(slice_scenario, W2Oracle())
    other = next(
        s for s in all_frozen
        if s.branch != slice_scenario.branch and s.required_decision == "escalate"
    )
    with pytest.raises(ReplayDivergence):
        replay(other, episode.trace)


def test_chain_detects_tampering(slice_scenario):
    """The hash chain is what makes a submitted trace evidence rather than a claim."""
    from cerl.core import ChainBroken

    episode = run_reference(slice_scenario, W2Oracle())
    episode.trace.verify_chain()  # the genuine article verifies

    entries = list(episode.trace.entries)
    victim = entries[2]
    entries[2] = victim.model_copy(
        update={"result": victim.result.model_copy(update={"message": "edited after the fact"})},
    )
    tampered = episode.trace.model_copy(update={"entries": tuple(entries)})
    with pytest.raises(ChainBroken):
        tampered.verify_chain()


def test_gold_trajectories_replay_to_their_recorded_hashes():
    for path in sorted(GOLD_DIR.glob("*.json")):
        gold = load_gold(path)
        scenario = freeze_module.load(
            Path("scenarios/frozen") / f"{gold.scenario_id}.json",
        )
        from cerl.reference import run_actions

        episode = run_actions(scenario, gold.actions)
        assert episode.final.state_hash() == gold.terminal_state_hash
        assert episode.trace.head_hash == gold.trace_head_hash


# --------------------------------------------------------------------------
# R2 - no ambient nondeterminism
# --------------------------------------------------------------------------


def test_no_wallclock_or_random_in_src():
    banned = ("datetime.now(", "time.time(", "time.monotonic(", "uuid4(", "random.")
    offenders = []
    for path in (REPO / "src" / "cerl").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert not offenders, "ambient nondeterminism in src/:\n" + "\n".join(offenders)


# Every symbol that could inject ambient nondeterminism, patched individually so
# a failure names the exact API that was reached rather than "something".
PATCHED_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("time", "time"),
    ("time", "time_ns"),
    ("time", "monotonic"),
    ("time", "monotonic_ns"),
    ("time", "perf_counter"),
    ("time", "perf_counter_ns"),
    ("time", "process_time"),
    ("time", "localtime"),
    ("time", "gmtime"),
    ("random", "random"),
    ("random", "randint"),
    ("random", "randrange"),
    ("random", "choice"),
    ("random", "choices"),
    ("random", "shuffle"),
    ("random", "sample"),
    ("random", "uniform"),
    ("random", "getrandbits"),
    ("random", "seed"),
    ("uuid", "uuid1"),
    ("uuid", "uuid3"),
    ("uuid", "uuid4"),
    ("uuid", "uuid5"),
    ("uuid", "getnode"),
    ("os", "urandom"),
    ("secrets", "token_bytes"),
    ("secrets", "token_hex"),
    ("secrets", "randbelow"),
    ("secrets", "choice"),
    ("datetime", "datetime"),
    ("datetime", "date"),
)


class AmbientNondeterminism(AssertionError):
    """Raised when the environment reaches for a nondeterministic source."""


def test_full_oracle_suite_runs_with_every_ambient_source_disabled(monkeypatch, all_frozen):
    """R2, behaviourally.

    Each of the symbols in ``PATCHED_SYMBOLS`` is replaced individually with a
    raising stub, then the complete oracle suite is run over every frozen
    scenario. Anything that reads a clock, draws a random number or mints a uuid
    fails loudly and names the symbol it reached for.

    ``datetime.datetime`` and ``datetime.date`` are replaced with subclasses whose
    constructors and ``now``/``today`` classmethods raise, because the C types
    themselves cannot be monkeypatched attribute-by-attribute.
    """
    import datetime as datetime_module
    import os as os_module
    import random as random_module
    import secrets as secrets_module
    import time as time_module
    import uuid as uuid_module

    modules = {
        "time": time_module,
        "random": random_module,
        "uuid": uuid_module,
        "os": os_module,
        "secrets": secrets_module,
        "datetime": datetime_module,
    }

    def exploding(qualified: str):
        def _raise(*_args: object, **_kwargs: object) -> None:
            raise AmbientNondeterminism(qualified)

        return _raise

    def forbidden_type(base: type, qualified: str) -> type:
        """A datetime/date subclass whose every entry point names *its own* symbol.

        Built in a factory rather than inline in the loop so each class closes
        over its own ``qualified``; a late-bound closure would report the last
        symbol in the loop and defeat the point of naming the exact API.
        """
        blow_up = exploding(qualified)

        return type(
            f"Forbidden{base.__name__.title()}",
            (base,),
            {
                "__new__": lambda _cls, *a, **k: blow_up(*a, **k),
                "now": classmethod(lambda _cls, *a, **k: exploding(f"{qualified}.now")()),
                "utcnow": classmethod(lambda _cls, *a, **k: exploding(f"{qualified}.utcnow")()),
                "today": classmethod(lambda _cls, *a, **k: exploding(f"{qualified}.today")()),
                "fromtimestamp": classmethod(
                    lambda _cls, *a, **k: exploding(f"{qualified}.fromtimestamp")(),
                ),
            },
        )

    patched: list[str] = []
    for module_name, attribute in PATCHED_SYMBOLS:
        module = modules[module_name]
        qualified = f"{module_name}.{attribute}"
        if module_name == "datetime":
            replacement: object = forbidden_type(getattr(module, attribute), qualified)
        else:
            replacement = exploding(qualified)
        monkeypatch.setattr(module, attribute, replacement, raising=False)
        patched.append(qualified)

    assert len(patched) == len(PATCHED_SYMBOLS)

    # The complete oracle suite, over every frozen scenario, with all of the
    # above disabled.
    for scenario in all_frozen:
        episode = run_reference(scenario, oracle_for(scenario))
        assert episode.verdict.is_clean_oracle_run, scenario.scenario_id
        replay(scenario, episode.trace)


def test_the_ambient_guard_actually_fires():
    """The premise of the test above, asserted rather than assumed."""
    import time as time_module

    original = time_module.time
    try:
        def _raise(*_args, **_kwargs):
            raise AmbientNondeterminism("time.time")

        time_module.time = _raise  # type: ignore[assignment]
        with pytest.raises(AmbientNondeterminism):
            time_module.time()
    finally:
        time_module.time = original  # type: ignore[assignment]


# --------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------


def test_episode_completes_within_the_env_time_budget(all_frozen):
    """The 50 ms/episode budget, measured without instrumentation.

    Coverage tracing multiplies wall time several-fold, so running this under
    ``--cov`` would measure the tracer rather than the environment. Skipping is
    honest; loosening the threshold until it passes under coverage would make the
    budget meaningless.
    """
    import sys
    import time

    if sys.gettrace() is not None:
        pytest.skip("timing is not meaningful under a tracer (coverage/debugger)")

    scenario = max(all_frozen, key=lambda s: len(s.world.billing.charges))
    for _ in range(3):
        run_reference(scenario, oracle_for(scenario))  # warm import and validation caches

    # Minimum of several batches rather than a single mean: inside a full test
    # session the mean picks up GC pauses and scheduler noise from neighbouring
    # tests, and the question here is what an episode costs, not what the
    # machine was doing at the time.
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


def test_reset_restores_the_exact_initial_state(all_frozen):
    scenario = all_frozen[0]
    env = CerlEnv(scenario)
    env.reset()
    before = env.world.state_hash()
    env.step(TicketsGet(ticket_id=TicketId(str(scenario.variables["ticket"]))))
    assert env.world.state_hash() != before
    env.reset()
    assert env.world.state_hash() == before
