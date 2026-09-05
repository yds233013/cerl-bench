"""The six safety-constraint classes.

Domain vocabulary rather than state, so it lives in ``core``: ``trace`` needs it
to tag each entry, and ``state`` needs it for the violation log. Keeping it in
``state`` created a ``trace <-> state`` package cycle, which is precisely what
prevented the architecture from being expressible as a layer stack.

Each class carries an independent *committed* and *attempted* series; the two are
never merged (CLAUDE.md rule 2).
"""

from __future__ import annotations

from enum import StrEnum


class ConstraintClass(StrEnum):
    AUTH = "C_AUTH"
    ENTITY = "C_ENTITY"
    IRREV = "C_IRREV"
    AMOUNT = "C_AMOUNT"
    DUP = "C_DUP"
    DISCLOSE = "C_DISCLOSE"


CONSTRAINT_ORDER: tuple[ConstraintClass, ...] = (
    ConstraintClass.AUTH,
    ConstraintClass.ENTITY,
    ConstraintClass.IRREV,
    ConstraintClass.AMOUNT,
    ConstraintClass.DUP,
    ConstraintClass.DISCLOSE,
)
