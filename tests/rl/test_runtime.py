"""R7: the deadline must bound the run, and records must survive interruption.

All synthetic: a fake clock, fake work, no model and no real delay. The point of
a bounded run is what happens when something overruns, and that is precisely
what cannot be tested by letting it run normally.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from cerl_rl.runtime import (
    Deadline,
    DeadlineExceeded,
    prepare_run_directory,
    write_atomic,
)


class FakeClock:
    """A clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------
# the deadline
# --------------------------------------------------------------------------


def test_the_deadline_is_monotonic_not_wall_clock():
    """``time.time`` can step backwards over an NTP correction; a budget must not."""
    import time

    assert Deadline(1).__init__.__defaults__ is None  # clock is keyword-only
    d = Deadline(10)
    assert d._clock is time.monotonic


def test_the_reproduced_overrun_is_now_refused():
    """The review's synthetic-clock case: zero updates, a 60s budget, and each
    fake evaluation advancing 100s. Previously the after-evaluation started
    after expiry and the run exited successfully recording 200 seconds."""
    clock = FakeClock()
    deadline = Deadline(60, clock=clock)

    deadline.enforce("baseline evaluation")      # fine at t=0
    clock.advance(100)                            # a phase overruns
    with pytest.raises(DeadlineExceeded, match="after evaluation"):
        deadline.enforce("after evaluation")


def test_the_reserve_is_actually_enforced_not_merely_reserved():
    """v1 subtracted a reserve when deciding whether to start another group and
    then ran the reserved phase unbounded anyway. Here the reserve is a real
    stopping condition: with 400s left and 480s reserved, training must stop."""
    clock = FakeClock()
    deadline = Deadline(600, clock=clock)
    clock.advance(100)
    deadline.enforce("training", reserve=480)     # 500 left, 480 reserved -> continue
    clock.advance(100)
    with pytest.raises(DeadlineExceeded, match="reserving 480s"):
        deadline.enforce("training", reserve=480)  # 400 left -> stop


def test_the_deadline_is_checked_inside_a_phase_not_only_between_phases():
    """Simulates an evaluation of several scenarios where each one overruns."""
    clock = FakeClock()
    deadline = Deadline(250, clock=clock)
    def run_five() -> int:
        done = 0
        for _ in range(5):
            deadline.enforce("evaluation")
            clock.advance(100)
            done += 1
        return done

    with pytest.raises(DeadlineExceeded):
        run_five()
    assert clock.now == 300, "the phase must stop mid-way, not run all five"


# --------------------------------------------------------------------------
# durability
# --------------------------------------------------------------------------


def test_a_record_write_cannot_destroy_the_previous_one(tmp_path):
    target = tmp_path / "pilot_run.json"
    write_atomic(target, {"groups": [1]})
    write_atomic(target, {"groups": [1, 2]})
    assert json.loads(target.read_text()) == {"groups": [1, 2]}


def test_a_failed_write_leaves_the_previous_record_intact(tmp_path, monkeypatch):
    """An interruption part-way through a write must not cost the last good record."""
    target = tmp_path / "pilot_run.json"
    write_atomic(target, {"good": True})

    from cerl_rl import runtime

    def explode(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime.json, "dump", explode)
    with pytest.raises(KeyboardInterrupt):
        write_atomic(target, {"replacement": True})

    assert json.loads(target.read_text()) == {"good": True}
    assert [p.name for p in tmp_path.iterdir()] == ["pilot_run.json"], (
        "a scratch file was left behind"
    )


def test_writing_into_a_directory_holding_a_run_is_refused(tmp_path):
    run = prepare_run_directory(tmp_path / "run")
    (run / "pilot_run.json").write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_run_directory(run)


def test_reuse_is_possible_but_only_deliberately(tmp_path):
    run = prepare_run_directory(tmp_path / "run")
    (run / "pilot_run.json").write_text("{}")
    assert prepare_run_directory(run, allow_existing=True) == run


def test_a_rerun_cannot_overwrite_the_recorded_pilot():
    """The historical run lives in evidence/rl-pilot, which was also the default
    output path. Whatever the default, that directory must now refuse reuse."""
    recorded = pathlib.Path(__file__).resolve().parents[2] / "evidence" / "rl-pilot"
    if not (recorded / "pilot_run.json").exists():
        pytest.skip("no recorded pilot in this checkout")
    with pytest.raises(FileExistsError):
        prepare_run_directory(recorded)
