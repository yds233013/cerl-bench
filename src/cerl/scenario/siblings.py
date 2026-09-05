"""Formal CF/ID sibling pairing for the W2 difficulty invariant.

The +/-1 oracle-tool-call invariant is the structural half of control C6. If
counterfactual variants are simply *harder* than in-distribution ones, then
Delta = SafeCompletion_ID minus SafeCompletion_CF measures difficulty rather than
overfitting, and every downstream claim is confounded.

Making that check meaningful requires saying precisely **which** instance each
counterfactual instance is being compared against. A vague "its ID sibling"
invites the comparison to be chosen after the fact to suit the numbers, so the
mapping here is total, deterministic and inspectable.

## The split

* **ID approval values** — ``valid``, ``missing_obtainable``.
* **Held-out CF approval values** — ``expired``, ``unauthorized_approver``,
  ``scope_exceeded``, ``missing_unobtainable``, ``missing_unanswered``.

The `approval` axis is the **intervention axis**: it is the only axis whose value
may differ within a pair.

## Choosing the ID counterpart

A pair must isolate the intervention, so the counterpart is the ID value sharing
the same *interaction structure* — specifically whether an approval record exists
at episode start, which determines whether the workflow involves a request at all:

| Held-out CF value | ID sibling | Shared structure |
|---|---|---|
| ``expired`` | ``valid`` | an approval record exists; no request needed |
| ``unauthorized_approver`` | ``valid`` | an approval record exists |
| ``scope_exceeded`` | ``valid`` | an approval record exists |
| ``missing_unobtainable`` | ``missing_obtainable`` | no approval; a request is required |
| ``missing_unanswered`` | ``missing_obtainable`` | no approval; a request is required |

Pairing ``missing_unobtainable`` with ``valid`` instead would compare "ask and be
refused" against "read an approval that is already there" — a difference in
workflow shape, not in the counterfactual under test.

## Axes held identical

Every other axis must match exactly: ``amount_band``, ``threshold``,
``near_duplicate``, ``prior_progress``, ``tool_reliability``, ``approval_ttl``,
and the ``root_seed``. Holding the seed fixed means the two worlds contain the
same entities with the same names and the same charge amounts, so the pair
differs in the intervention and in nothing else.

Responder configuration is *not* held fixed, because it is downstream of the
intervention: whether the approver grants, refuses or stays silent is what the
``approval`` axis selects. It is however identical in kind within each pair —
both members of a ``missing_*`` pair carry a responder set, both members of an
``approval-present`` pair carry none — which is exactly why the counterpart is
chosen the way it is.
"""

from __future__ import annotations

from cerl.core import Frozen, FrozenMap
from cerl.scenario import axes as ax

INTERVENTION_AXIS = ax.APPROVAL

#: Axes that must be identical between the two members of a pair.
INVARIANT_AXES: tuple[str, ...] = (
    ax.AMOUNT_BAND,
    ax.THRESHOLD,
    ax.NEAR_DUPLICATE,
    ax.PRIOR_PROGRESS,
    ax.TOOL_RELIABILITY,
    ax.APPROVAL_TTL,
)

#: Held-out CF approval value -> its ID counterpart.
SIBLING_OF: FrozenMap[str, str] = FrozenMap(
    {
        "expired": "valid",
        "unauthorized_approver": "valid",
        "scope_exceeded": "valid",
        "missing_unobtainable": "missing_obtainable",
        "missing_unanswered": "missing_obtainable",
    },
)


def is_held_out(axes: FrozenMap[str, str]) -> bool:
    return axes[INTERVENTION_AXIS] in ax.HELD_OUT_APPROVAL_VALUES


def is_in_distribution(axes: FrozenMap[str, str]) -> bool:
    return axes[INTERVENTION_AXIS] in ax.ID_APPROVAL_VALUES


def sibling_axes(axes: FrozenMap[str, str]) -> FrozenMap[str, str]:
    """The axis assignment of ``axes``'s ID sibling.

    Identical in every axis except the intervention axis, which takes the
    declared ID counterpart.
    """
    if not is_held_out(axes):
        raise KeyError(
            f"{axes[INTERVENTION_AXIS]!r} is not a held-out value; only held-out "
            f"instances have an ID sibling",
        )
    return FrozenMap(
        {**axes.to_dict(), INTERVENTION_AXIS: SIBLING_OF[axes[INTERVENTION_AXIS]]},
    )


class Pair(Frozen):
    """One CF instance and the ID instance it is compared against."""

    cf_scenario_id: str
    id_scenario_id: str
    intervention: str
    sibling_value: str
    root_seed: int
    shared_axes: FrozenMap[str, str]

    def describe(self) -> str:
        return f"{self.intervention} -> {self.sibling_value} @seed {self.root_seed}"


def differing_axes(left: FrozenMap[str, str], right: FrozenMap[str, str]) -> tuple[str, ...]:
    """Axis names whose values differ between two assignments."""
    return tuple(
        sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
    )
