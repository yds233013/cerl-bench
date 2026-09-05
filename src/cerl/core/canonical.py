"""Canonical JSON serialization and content hashing.

Every hash in the project flows through :func:`canonical_json`. The encoding is
deliberately narrow so that two runs, two processes, and two Python minor
versions cannot disagree:

* object keys sorted (byte order of the UTF-8 key)
* no insignificant whitespace
* ``ensure_ascii=False`` with UTF-8 output
* **floats are rejected** -- money is integer cents and state carries no floats,
  so a float reaching here is a bug, not a value to round

See CLAUDE.md rule 1.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cerl.core.errors import NonCanonicalValue

_HASH_DIGEST_SIZE = 32


def _reject_floats(value: Any) -> Any:
    """Recursively assert no float appears in ``value``.

    ``json.dumps`` would happily emit ``1.1``, whose repr is platform-stable in
    CPython but whose *presence* in state means someone modelled money or time
    as a float. That is the defect we want surfaced, so we raise.
    """
    if isinstance(value, float):
        raise NonCanonicalValue(f"float {value!r} is not permitted in canonical JSON")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NonCanonicalValue(f"non-string object key {key!r}")
            _reject_floats(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_floats(item)
    return value


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text for ``value``."""
    _reject_floats(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """Return canonical JSON encoded as UTF-8."""
    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    """Return the blake2b hex digest of ``value``'s canonical encoding."""
    return hashlib.blake2b(canonical_bytes(value), digest_size=_HASH_DIGEST_SIZE).hexdigest()


def _fixed_precision(value: Any) -> Any:
    """Render floats at fixed precision so a report can be hashed stably."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, dict):
        return {key: _fixed_precision(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fixed_precision(item) for item in value]
    return value


def report_hash(value: Any) -> str:
    """Content hash for an *output report* rather than for state.

    State forbids floats outright, because an exact comparison over one is a
    latent nondeterminism bug. A metrics block is different: rates are genuinely
    fractional and are read by people. Rendering them at fixed precision before
    hashing keeps the artifact tamper-evident without smuggling floats into the
    state discipline.
    """
    return content_hash(_fixed_precision(value))


def hash_text(text: str) -> str:
    """Return the blake2b hex digest of raw text (for file/manifest hashing)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=_HASH_DIGEST_SIZE).hexdigest()
