"""UNPRIVILEGED policies only.

``agents`` may not import ``reference``, ``verify`` or ``scenario``
(import-linter contract 1). Phase 1A ships no evaluated agent -- only the
scripted helper used by adversarial and golden fixtures.
"""

from cerl.agents.base import Agent, ScriptedAgent, is_unprivileged_agent

__all__ = ["Agent", "ScriptedAgent", "is_unprivileged_agent"]
