"""Deadline enforcement and durable record-keeping for a bounded run.

Three defects the first version had, each of which only shows up when something
goes wrong -- which is exactly when a bounded run needs to behave:

* the deadline used ``time.time`` (wall clock, which can step backwards over an
  NTP correction or a DST change) and was consulted **only between groups**, so
  baseline evaluation, every generation inside a group, each backward pass and
  the whole after-evaluation ran unbounded. Reserving eight minutes reserved
  nothing;
* records were written by truncating the destination in place, so an
  interruption mid-write left a corrupt file and lost the previous one;
* the default output path was the directory holding the original evidence, so a
  rerun would overwrite recorded results.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
from typing import Any


class DeadlineExceeded(RuntimeError):
    """The run's time budget is spent. Raised so callers unwind and save."""


class Deadline:
    """A monotonic budget, checkable at any granularity.

    ``time.monotonic`` cannot go backwards, which ``time.time`` can. It is
    checked *inside* phases as well as between them, and :meth:`enforce` raises
    rather than returning a flag a caller can forget to read.
    """

    def __init__(self, seconds: float, *, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self.limit = float(seconds)
        self.started = self._clock()

    @property
    def elapsed(self) -> float:
        return self._clock() - self.started

    @property
    def remaining(self) -> float:
        return self.limit - self.elapsed

    def expired(self, reserve: float = 0.0) -> bool:
        return self.remaining <= reserve

    def enforce(self, phase: str, reserve: float = 0.0) -> None:
        """Raise if the budget (less ``reserve``) is spent."""
        if self.expired(reserve):
            raise DeadlineExceeded(
                f"{phase}: {self.elapsed:.1f}s elapsed of a {self.limit:.1f}s budget"
                + (f" (reserving {reserve:.0f}s)" if reserve else ""),
            )


def write_atomic(path: pathlib.Path, payload: Any) -> None:
    """Write JSON so an interruption cannot destroy the previous version.

    Written to a temporary file in the same directory, flushed, ``fsync``ed and
    then renamed. ``os.replace`` is atomic within a filesystem, so a reader sees
    either the old file or the new one -- never a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    scratch = pathlib.Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(payload, file, indent=2, default=str)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        scratch.replace(path)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise


def prepare_run_directory(
    path: pathlib.Path, *, allow_existing: bool = False,
) -> pathlib.Path:
    """Refuse to write into a directory that already holds a run.

    The first version defaulted to the directory containing the recorded pilot,
    so an accidental rerun would have overwritten the evidence it was supposed
    to be compared against.
    """
    existing = path / "pilot_run.json"
    if existing.exists() and not allow_existing:
        raise FileExistsError(
            f"{existing} already exists; refusing to overwrite a recorded run. "
            f"Choose a new --out directory, or pass allow_existing to reuse it.",
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
