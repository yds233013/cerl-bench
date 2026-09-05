"""Project exception hierarchy.

Domain nouns, not ``*Error`` suffixes (see ruff N818 ignore in pyproject).
"""

from __future__ import annotations


class CerlFault(Exception):
    """Base for every CERL-Bench fault."""


class InvalidEntityId(CerlFault, ValueError):
    """An entity identifier did not match its type's frozen pattern.

    Also a ``ValueError`` so Pydantic wraps it in a ``ValidationError`` during
    model validation: a swapped id in a frozen scenario should surface as an
    ordinary schema failure at load time, with the field path attached, rather
    than as an opaque exception escaping the parser.
    """

    def __init__(self, type_name: str, raw: str) -> None:
        super().__init__(f"{type_name} rejects {raw!r}: prefix/format mismatch")
        self.type_name = type_name
        self.raw = raw


class NonCanonicalValue(CerlFault):
    """A value cannot be canonically serialized (e.g. a float in state)."""


class FrozenMapMutation(CerlFault):
    """Attempted mutation of a FrozenMap."""


class PrivilegeViolation(CerlFault):
    """Attempted construction of privileged ground truth outside reference/."""


class ScenarioDefect(CerlFault):
    """A scenario is internally inconsistent (branches, rubric, or allowlist)."""


class ChainBroken(CerlFault):
    """A trace's hash chain failed verification."""


class ReplayDivergence(CerlFault):
    """Replaying a trace did not reproduce it byte-for-byte."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"replay diverged: {detail}")
        self.detail = detail
