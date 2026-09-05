"""What the agent sees.

The observation derives **only** from tool returns. There is no global state
view, no convenience field, no rubric hint, no axis label, and -- deliberately --
no notification that a responder fired. If an approval arrives, the agent must
discover it by reading the thread, exactly as a person would. Any information the
agent needs costs a tool call, which is what makes tool-call efficiency
meaningful and "the agent failed to check the approval" a real failure rather
than an artifact of us handing it the approval.
"""

from __future__ import annotations

from cerl.actions import ToolResult
from cerl.core import Frozen, LogicalInstant


class TaskBrief(Frozen):
    """Step-0 framing. Says what the workflow is, never which variant it is."""

    text: str
    ticket_id: str
    budget_steps: int
    agent_user: str
    available_tools: tuple[str, ...]


class Notice(Frozen):
    code: str
    text: str


class Observation(Frozen):
    brief: TaskBrief | None = None
    result: ToolResult | None = None
    step_index: int
    steps_remaining: int
    # Exposed so the agent can reason about expiry. Without it the approval axis
    # would test perception rather than compliance.
    logical_time: LogicalInstant
    notices: tuple[Notice, ...] = ()
