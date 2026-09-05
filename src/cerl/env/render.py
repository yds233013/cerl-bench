"""Render an observation as agent-facing text.

Versioned, because prompt formatting is a confound that must be held fixed
across arms in any comparison. ``renderer_version`` is recorded in run manifests
so two results produced under different renderings can never be silently
compared.

This function is also the surface that ``tests/boundaries`` scans for ground
truth leakage: it checks the *rendered string*, because the type system cannot
tell you that a payload field happens to name the resolved branch.
"""

from __future__ import annotations

import json

from cerl.env.observation import Observation

RENDERER_VERSION = "1.0.0"


def render_observation(observation: Observation) -> str:
    lines: list[str] = []
    if observation.brief is not None:
        brief = observation.brief
        lines += [
            "# Task",
            brief.text,
            "",
            f"Ticket: {brief.ticket_id}",
            f"You are acting as: {brief.agent_user}",
            f"Step budget: {brief.budget_steps}",
            "",
            "# Tools",
            ", ".join(brief.available_tools),
            "",
        ]

    lines.append(
        f"# Step {observation.step_index} "
        f"({observation.steps_remaining} remaining, logical time {int(observation.logical_time)})",
    )

    if observation.result is not None:
        result = observation.result
        lines.append(f"outcome: {result.outcome}")
        if result.message:
            lines.append(f"message: {result.message}")
        if result.denied_interlock:
            lines.append(f"refused_by: {result.denied_interlock}")
        if result.payload:
            lines.append("payload:")
            lines.append(json.dumps(result.payload.to_dict(), indent=2, sort_keys=True))

    for notice in observation.notices:
        lines.append(f"notice[{notice.code}]: {notice.text}")

    return "\n".join(lines)
