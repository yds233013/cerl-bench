"""Rendering the **complete** public tool contract for the policy prompt.

The first version showed names and argument names only::

    - tickets__set_status(status, ticket_id)

which withholds most of what the core already publishes: the description, each
argument's type, which arguments are required, the permitted values of an enum,
numeric bounds, and whether unknown arguments are accepted. The saved pilot
evidence contains four malformed attempts to set a ticket's status to ``closed``
-- a value the schema excludes and the prompt never showed.

That does not mean showing the schema would have made the model succeed. It
means a failure to use an interface the policy was never shown cannot be read as
inability to follow a specified one, so the abbreviated menu made those failures
uninterpretable.

Everything rendered here is already public: it is the same
``all_tool_schemas()`` an ordinary evaluated agent receives. No branch, rubric,
expected action or other privileged material passes through this module -- the
schemas describe what *can* be asked for, never what is correct.
"""

from __future__ import annotations

from typing import Any

from cerl.agents.tool_schemas import all_tool_schemas


def _constraint_phrases(spec: dict[str, Any]) -> list[str]:
    """Human-readable form of the constraints the schema actually carries."""
    out: list[str] = []
    if "enum" in spec:
        out.append("one of " + "|".join(str(v) for v in spec["enum"]))
    for key, phrase in (
        ("exclusiveMinimum", "> {}"),
        ("minimum", ">= {}"),
        ("exclusiveMaximum", "< {}"),
        ("maximum", "<= {}"),
    ):
        if key in spec:
            out.append(phrase.format(spec[key]))
    return out


def render_argument(name: str, spec: dict[str, Any], *, required: bool) -> str:
    parts = [f"{name}: {spec.get('type', 'any')}"]
    parts.extend(_constraint_phrases(spec))
    parts.append("required" if required else "optional")
    return f"      {', '.join(parts)}"


def render_tool(schema: dict[str, Any]) -> list[str]:
    inner = schema["input_schema"]
    properties: dict[str, Any] = inner.get("properties", {})
    required = set(inner.get("required", []))
    lines = [f"  {schema['name']}", f"    {schema.get('description', '').strip()}"]
    if properties:
        lines.append("    arguments:")
        lines.extend(
            render_argument(name, spec, required=name in required)
            for name, spec in sorted(properties.items())
        )
    else:
        lines.append("    arguments: none")
    if inner.get("additionalProperties") is False:
        lines.append("    no other arguments are accepted")
    return lines


def tool_contract() -> str:
    """The full advertised contract, in the order the schemas are published."""
    lines: list[str] = []
    for schema in all_tool_schemas():
        lines.extend(render_tool(schema))
        lines.append("")
    return "\n".join(lines).rstrip()


def tool_menu_v1() -> str:
    """The abbreviated menu the historical pilot used. Kept only so the recorded
    run stays reproducible; it is not what a corrected run should show."""
    lines = []
    for schema in all_tool_schemas():
        args = ", ".join(sorted(schema["input_schema"]["properties"]))
        lines.append(f"- {schema['name']}({args})")
    return "\n".join(lines)
