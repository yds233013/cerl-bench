"""RFC 6901 JSON pointers and the pattern language used by allowlists.

A pattern is a pointer whose segments may be:

* a literal segment              -- ``billing``
* ``*``                          -- exactly one segment, any value
* ``$.name``                     -- a scenario variable, substituted before match

Patterns match a *prefix or the whole* of an op path: an allowlist entry for
``/billing/refunds/*`` permits the whole added refund object at that path and any
field beneath it, which is what makes value-constrained subtree permits work.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WILDCARD = "*"
VAR_PREFIX = "$."


def escape(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def unescape(segment: str) -> str:
    return segment.replace("~1", "/").replace("~0", "~")


def join(*segments: str | int) -> str:
    return "".join(f"/{escape(str(s))}" for s in segments)


def split(pointer: str) -> tuple[str, ...]:
    if pointer in {"", "/"}:
        return ()
    if not pointer.startswith("/"):
        raise ValueError(f"pointer must start with '/': {pointer!r}")
    return tuple(unescape(s) for s in pointer[1:].split("/"))


def resolve_vars(pattern: str, variables: Mapping[str, Any]) -> str:
    """Substitute ``$.name`` segments using ``variables``.

    An unresolvable variable is left as-is; it then simply fails to match, which
    surfaces as a residual rather than as a silent permit. Failing closed is the
    right default for an allowlist.
    """
    out: list[str] = []
    for segment in split(pattern):
        if segment.startswith(VAR_PREFIX):
            name = segment[len(VAR_PREFIX) :]
            if name in variables:
                out.append(str(variables[name]))
                continue
        out.append(segment)
    return join(*out)


def matches(pattern: str, pointer: str, variables: Mapping[str, Any] | None = None) -> bool:
    """Return True if ``pointer`` is at or beneath ``pattern``."""
    resolved = resolve_vars(pattern, variables or {})
    pat = split(resolved)
    ptr = split(pointer)
    if len(ptr) < len(pat):
        return False
    return all(p in {WILDCARD, q} for p, q in zip(pat, ptr, strict=False))


def get(document: Any, pointer: str) -> Any:
    """Read ``pointer`` out of a plain JSON document; None if absent."""
    node = document
    for segment in split(pointer):
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
        elif isinstance(node, (list, tuple)):
            try:
                node = node[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def get_field(document: Any, dotted: str) -> Any:
    """Read a dotted field path out of a nested JSON value."""
    node = document
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return None
    return node
