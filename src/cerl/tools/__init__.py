"""Typed tool interfaces.

``tools`` is the only package permitted to mutate ``WorldState``
(import-linter contract 6, test-enforced).
"""

from cerl.tools import interlocks
from cerl.tools.context import ToolContext, mint_runtime_id
from cerl.tools.interlocks import (
    INTERLOCK_IDS,
    INTERLOCKS,
    Interlock,
    attempted_class_for,
    interlock,
)
from cerl.tools.registry import (
    META_TICK_COST,
    REGISTRY,
    TOOL_COUNT,
    TOOL_TICK_COST,
    ToolHandler,
    handler_for,
    tick_cost,
)

__all__ = [
    "INTERLOCKS",
    "INTERLOCK_IDS",
    "META_TICK_COST",
    "REGISTRY",
    "TOOL_COUNT",
    "TOOL_TICK_COST",
    "Interlock",
    "ToolContext",
    "ToolHandler",
    "attempted_class_for",
    "handler_for",
    "interlock",
    "interlocks",
    "mint_runtime_id",
    "tick_cost",
]
