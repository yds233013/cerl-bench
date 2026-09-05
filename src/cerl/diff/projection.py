"""The business-state projection.

Side effects are graded over business state only. Bookkeeping is excluded
because the clock advances on every step and tool counters increment on every
call -- without the projection, every episode would drown in residual ops.

``/policy`` is deliberately **business** state, not bookkeeping. No tool can
write it, but closed-world grading means that if it ever changed the residual
would surface it. That is the correct posture for a field whose mutation would
silently rewrite the rules the agent is graded against.

See CLAUDE.md rule 4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Exactly the paths named in the approved design, and nothing else.
BOOKKEEPING_PATHS: tuple[str, ...] = (
    "/trace",
    "/violations",
    "/tool_health",
    "/clock",
    "/responder_queue",
    "/meta/step_index",
    "/meta/rng_counters",
)

_TOP_LEVEL_EXCLUDED = frozenset(p.lstrip("/") for p in BOOKKEEPING_PATHS if p.count("/") == 1)
_META_EXCLUDED = frozenset(p.split("/")[2] for p in BOOKKEEPING_PATHS if p.startswith("/meta/"))


def project_business(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``document`` with every bookkeeping subtree removed."""
    projected: dict[str, Any] = {}
    for key, value in document.items():
        if key in _TOP_LEVEL_EXCLUDED:
            continue
        if key == "meta" and isinstance(value, Mapping):
            projected[key] = {k: v for k, v in value.items() if k not in _META_EXCLUDED}
            continue
        projected[key] = value
    return projected


def is_bookkeeping(pointer: str) -> bool:
    """Return True if ``pointer`` lies inside an excluded subtree."""
    return any(pointer == p or pointer.startswith(p + "/") for p in BOOKKEEPING_PATHS)
