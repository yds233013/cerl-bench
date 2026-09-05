"""Structural diff over canonical JSON documents.

Collections keyed by id are modelled as JSON *objects*, never arrays, so
inserting a refund produces one ``add`` at ``/billing/refunds/re_...`` rather
than a cascade of index-shift ops. That property is what makes pointer-pattern
allowlists workable at all.

Arrays (ticket comments, mention lists) are compared as whole values: a change
anywhere inside produces a single ``replace`` at the array's own path. This is
deliberate -- an append-only comment list has no meaningful per-index identity,
and per-index ops would invite allowlists that permit "any comment index" while
silently permitting rewrites of earlier ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cerl.diff.models import DiffOp, DiffOpKind, Origin, StateDiff
from cerl.diff.pointer import join
from cerl.diff.projection import project_business

_MISSING = object()


def _walk(
    before: Any,
    after: Any,
    path: str,
    out: list[DiffOp],
    origin: Origin,
    responder_rule: str | None,
) -> None:
    if before is _MISSING:
        out.append(
            DiffOp(
                op=DiffOpKind.ADD,
                path=path,
                before=None,
                after=after,
                origin=origin,
                responder_rule=responder_rule,
            ),
        )
        return
    if after is _MISSING:
        out.append(
            DiffOp(
                op=DiffOpKind.REMOVE,
                path=path,
                before=before,
                after=None,
                origin=origin,
                responder_rule=responder_rule,
            ),
        )
        return

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after)):
            _walk(
                before.get(key, _MISSING),
                after.get(key, _MISSING),
                path + join(key),
                out,
                origin,
                responder_rule,
            )
        return

    if before != after:
        out.append(
            DiffOp(
                op=DiffOpKind.REPLACE,
                path=path,
                before=before,
                after=after,
                origin=origin,
                responder_rule=responder_rule,
            ),
        )


def diff_documents(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    origin: Origin = Origin.AGENT,
    responder_rule: str | None = None,
) -> StateDiff:
    """Diff two canonical JSON documents in full (no projection)."""
    ops: list[DiffOp] = []
    _walk(dict(before), dict(after), "", ops, origin, responder_rule)
    return StateDiff.of(tuple(ops))


def diff_business(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    origin: Origin = Origin.AGENT,
    responder_rule: str | None = None,
) -> StateDiff:
    """Diff two documents after applying the business projection."""
    return diff_documents(
        project_business(before),
        project_business(after),
        origin=origin,
        responder_rule=responder_rule,
    )


def apply_diff(document: Mapping[str, Any], diff: StateDiff) -> dict[str, Any]:
    """Apply ``diff`` to ``document``. Used by the round-trip property test."""
    from cerl.diff.pointer import split

    result: dict[str, Any] = _deep_copy(dict(document))
    for op in diff.ops:
        segments = split(op.path)
        if not segments:
            continue
        node: Any = result
        for segment in segments[:-1]:
            node = node.setdefault(segment, {})
        leaf = segments[-1]
        if op.op is DiffOpKind.REMOVE:
            node.pop(leaf, None)
        else:
            node[leaf] = _deep_copy(op.after)
    return result


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value
