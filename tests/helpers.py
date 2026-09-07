"""Shared test helpers reachable as a plain module (not via the conftest package)."""

from __future__ import annotations

import re
from pathlib import Path

from cerl.scenario import freeze as _freeze

REPO = Path(__file__).resolve().parents[1]
#: The canonical corpus (2.0.0). The 1.x corpus stays on disk under
#: ``scenarios/frozen`` as an immutable historical artifact.
FROZEN_DIR = REPO / "scenarios" / "v2" / "frozen"
GOLD_DIR = REPO / "scenarios" / "v2" / "gold"
LEGACY_FROZEN_DIR = REPO / "scenarios" / "frozen"
LEGACY_MANIFEST = REPO / "scenarios" / "manifest.json"


def frozen_paths() -> list[Path]:
    return sorted(FROZEN_DIR.glob("*.json"))


def load_all_frozen():
    return [_freeze.load(p) for p in frozen_paths()]


#: Matches an ANSI SGR escape sequence -- the colour and style codes Rich emits.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def visible_text(output: str) -> str:
    """CLI output as a person reads it: no styling, no wrap artifacts.

    Asserting a phrase against raw ``Result.output`` looks right and is not,
    because two things sit between the message the code wrote and the bytes the
    runner captured:

    * **Styling.** When colour is on -- which it is on GitHub Actions and is not
      in a plain local shell -- Rich's highlighter styles CLI options, so
      ``--ledger-out`` is emitted as three separately-coloured runs and the
      literal substring ``"fresh --ledger-out"`` is simply not there. The text is
      on the screen; it is not in the string.
    * **Wrapping.** Rich hard-wraps to the panel width, so any phrase can be
      split across a line break, and where the break lands depends on things
      like the length of a temporary path.

    Both make an assertion pass or fail for reasons that have nothing to do with
    what the CLI said, which is how a test can be green on every developer's
    machine and red on CI. Normalising here keeps the assertion checking the
    message and stops it checking the terminal.
    """
    return " ".join(_ANSI.sub("", output).split())
