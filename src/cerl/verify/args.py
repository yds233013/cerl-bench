"""Resolve ``$.name`` references in rubric/invariant arguments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$."):
        return variables.get(value[2:], value)
    if isinstance(value, (list, tuple)):
        return [resolve(v, variables) for v in value]
    if isinstance(value, Mapping):
        return {k: resolve(v, variables) for k, v in value.items()}
    return value


def resolve_args(args: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[str, Any]:
    return {key: resolve(value, variables) for key, value in args.items()}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]
