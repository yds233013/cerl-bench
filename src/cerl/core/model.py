"""The frozen base model shared by every data layer.

Lives in ``core`` so that ``actions``, ``diff``, ``trace`` and ``state`` can all
use it without an import cycle.

``frozen=True`` blocks attribute assignment; ``extra="forbid"`` turns a typo in a
scenario file into a load-time error rather than a silently ignored field. Note
that ``frozen`` is *shallow* in Pydantic -- deep immutability comes from using
only ``tuple`` / ``frozenset`` / ``FrozenMap`` for container fields, which
``tests/immutability`` audits reflectively (CLAUDE.md rule 8).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable, strict-schema base model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
