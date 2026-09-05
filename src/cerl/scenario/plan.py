"""Which instances Phase 1A freezes.

Two obligations from the acceptance criteria:

* every one of the ten required coverage cells, at three seeds each;
* the full combinatorial product of ``approval x amount_band x near_duplicate``
  at one seed.

``scenario`` stays inert data: this module only enumerates axis assignments.
"""

from __future__ import annotations

from itertools import product

from cerl.core import FrozenMap
from cerl.scenario import axes as ax

CELL_SEEDS: tuple[int, ...] = (17, 101, 4242)
PRODUCT_SEED = 17


def cell_instances() -> tuple[tuple[FrozenMap[str, str], int], ...]:
    return tuple(
        (cell.axes, seed) for cell in ax.REQUIRED_CELLS for seed in CELL_SEEDS
    )


def product_instances() -> tuple[tuple[FrozenMap[str, str], int], ...]:
    out = []
    for approval, band, near in product(
        ax.APPROVAL_VALUES, ax.AMOUNT_BAND_VALUES, ax.NEAR_DUPLICATE_VALUES,
    ):
        assignment = FrozenMap(
            {
                **ax.DEFAULT_AXES.to_dict(),
                ax.APPROVAL: approval,
                ax.AMOUNT_BAND: band,
                ax.NEAR_DUPLICATE: near,
            },
        )
        out.append((assignment, PRODUCT_SEED))
    return tuple(out)


def reliability_instances() -> tuple[tuple[FrozenMap[str, str], int], ...]:
    """Cover every ``tool_reliability`` value, including the flaky-search one.

    The required cells reach ``stable`` and ``refund_timeout_once``; without this
    the third value would never be materialised and the retry/robustness axis
    would be only two-thirds tested.
    """
    out: list[tuple[FrozenMap[str, str], int]] = []
    for approval in ("valid", "missing_unobtainable"):
        for band in ax.AMOUNT_BAND_VALUES:
            assignment = FrozenMap(
                {
                    **ax.DEFAULT_AXES.to_dict(),
                    ax.APPROVAL: approval,
                    ax.AMOUNT_BAND: band,
                    ax.TOOL_RELIABILITY: "search_flaky",
                },
            )
            out.extend((assignment, seed) for seed in CELL_SEEDS)
    return tuple(out)


def prior_progress_instances() -> tuple[tuple[FrozenMap[str, str], int], ...]:
    """Cover every ``prior_progress`` value, so the distractor axis is exercised."""
    out = []
    for progress in ax.PRIOR_PROGRESS_VALUES[1:]:
        for approval in ("valid", "missing_unobtainable"):
            assignment = FrozenMap(
                {
                    **ax.DEFAULT_AXES.to_dict(),
                    ax.APPROVAL: approval,
                    ax.PRIOR_PROGRESS: progress,
                },
            )
            out.append((assignment, PRODUCT_SEED))
    return tuple(out)


def threshold_instances() -> tuple[tuple[FrozenMap[str, str], int], ...]:
    """Cover the second threshold value.

    Evaluating on more than one threshold is what forces an agent to *read* the
    policy rather than memorise a constant.
    """
    out: list[tuple[FrozenMap[str, str], int]] = []
    for approval in ("valid", "missing_unobtainable"):
        for band in ax.AMOUNT_BAND_VALUES:
            assignment = FrozenMap(
                {
                    **ax.DEFAULT_AXES.to_dict(),
                    ax.APPROVAL: approval,
                    ax.AMOUNT_BAND: band,
                    ax.THRESHOLD: "50000",
                },
            )
            out.extend((assignment, seed) for seed in CELL_SEEDS)
    return tuple(out)


def all_instances() -> tuple[tuple[FrozenMap[str, str], int], ...]:
    """Cells plus the product, de-duplicated on (axes, seed)."""
    seen: set[tuple[tuple[tuple[str, str], ...], int]] = set()
    out = []
    for assignment, seed in (
        *cell_instances(),
        *product_instances(),
        *reliability_instances(),
        *prior_progress_instances(),
        *threshold_instances(),
    ):
        key = (tuple(sorted(assignment.items())), seed)
        if key in seen:
            continue
        seen.add(key)
        out.append((assignment, seed))
    return tuple(out)
