"""Tool execution context and deterministic runtime id minting."""

from __future__ import annotations

from typing import TypeVar

from cerl.core import EntityId, Frozen, UserId, content_hash
from cerl.state.world import WorldState

IdT = TypeVar("IdT", bound=EntityId)

_ID_SPACE = 1 << 48


class ToolContext(Frozen):
    """Who is acting. The environment stamps this; an agent cannot supply it."""

    actor: UserId


def mint_runtime_id(
    id_type: type[IdT],
    world: WorldState,
    discriminator: str,
    ordinal: int,
) -> IdT:
    """Deterministically mint an id for an entity created during an episode.

    Derived from ``(scenario_id, root_seed, discriminator, ordinal)`` where
    ``ordinal`` is how many entities of this kind already exist. Deliberately
    *not* keyed on the step index: an id that moves when an unrelated read is
    inserted makes any recorded trajectory fragile under edit, which breaks
    exactly the mutation and diversity testing the verifier depends on. Keyed on
    what exists rather than on when it was made, an id is stable under any
    insertion that does not itself create an entity.

    Still fully deterministic under replay, and cannot collide with
    generator-minted ids, which occupy the low index range. No ``uuid``, no
    ``random`` (CLAUDE.md rule 1).
    """
    digest = content_hash(
        {
            "scenario": world.meta.scenario_id,
            "seed": world.meta.root_seed,
            "kind": discriminator,
            "ordinal": ordinal,
        },
    )
    return id_type.mint(int(digest[:16], 16) % _ID_SPACE)
