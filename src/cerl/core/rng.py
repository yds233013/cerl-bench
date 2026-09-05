"""Counter-based, domain-separated key derivation.

    subkey = blake2b(root_seed || domain_label || counter)

There is no mutable RNG stream anywhere. This is consequential, not stylistic:
with a shared stream, an agent making one extra read call would shift every
downstream draw, so two policies would face different worlds and could not be
compared. Counter-based derivation makes every generated value a stable
property of ``(root_seed, domain, counter)``.

Used only by the scenario generator. At episode time the world is fully
materialized and tool failures come from an explicit frozen schedule, so the
environment itself draws no randomness at all.

See CLAUDE.md rule 1.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")

_SUBKEY_SIZE = 16


def derive_bytes(root_seed: int, domain: str, counter: int) -> bytes:
    """Return 16 deterministic bytes for ``(root_seed, domain, counter)``."""
    payload = b"|".join(
        (
            str(root_seed).encode("ascii"),
            domain.encode("utf-8"),
            str(counter).encode("ascii"),
        ),
    )
    return hashlib.blake2b(payload, digest_size=_SUBKEY_SIZE).digest()


def derive_int(root_seed: int, domain: str, counter: int) -> int:
    """Return a non-negative deterministic integer."""
    return int.from_bytes(derive_bytes(root_seed, domain, counter), "big")


def derive_below(root_seed: int, domain: str, counter: int, bound: int) -> int:
    """Return a deterministic integer in ``[0, bound)``."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    return derive_int(root_seed, domain, counter) % bound


def derive_range(root_seed: int, domain: str, counter: int, low: int, high: int) -> int:
    """Return a deterministic integer in ``[low, high]`` inclusive."""
    if high < low:
        raise ValueError("high must be >= low")
    return low + derive_below(root_seed, domain, counter, high - low + 1)


def derive_choice(root_seed: int, domain: str, counter: int, options: Sequence[T]) -> T:
    """Return a deterministic element of ``options``."""
    if not options:
        raise ValueError("options must be non-empty")
    return options[derive_below(root_seed, domain, counter, len(options))]


class KeyedRng:
    """A convenience cursor over one derivation domain.

    Holds an explicit integer counter rather than internal entropy, so the value
    at any position is reproducible in isolation: ``KeyedRng(s, d).at(7)`` is
    the same number no matter what was drawn before it.
    """

    __slots__ = ("_counter", "_domain", "_root_seed")

    def __init__(self, root_seed: int, domain: str) -> None:
        self._root_seed = root_seed
        self._domain = domain
        self._counter = 0

    @property
    def counter(self) -> int:
        return self._counter

    def _next(self) -> int:
        value = self._counter
        self._counter += 1
        return value

    def at(self, counter: int) -> int:
        return derive_int(self._root_seed, self._domain, counter)

    def below(self, bound: int) -> int:
        return derive_below(self._root_seed, self._domain, self._next(), bound)

    def between(self, low: int, high: int) -> int:
        return derive_range(self._root_seed, self._domain, self._next(), low, high)

    def choice(self, options: Sequence[T]) -> T:
        return derive_choice(self._root_seed, self._domain, self._next(), options)

    def sub(self, label: str) -> KeyedRng:
        """Return an independent cursor in a nested domain."""
        return KeyedRng(self._root_seed, f"{self._domain}/{label}")
