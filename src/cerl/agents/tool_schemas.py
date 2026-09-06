"""Public tool schemas for an evaluated agent.

Generated from the action models, which are public by construction: they are the
same definitions the environment validates against, so a schema cannot drift
from what the tool actually accepts.

Nothing privileged reaches here. The schemas describe *what can be asked for*,
never what the right answer is -- no branch names, no axis values, no rubric
predicates, no expected actions. ``tests/boundaries`` scans the rendered result.
"""

from __future__ import annotations

from typing import Any

from cerl.actions import ACTION_TYPES, ActionKind, BaseAction
from cerl.actions.models import MalformedAction

#: Human-readable purpose for each tool, written for an agent that has never
#: seen this environment. Deliberately free of outcome hints.
DESCRIPTIONS: dict[str, str] = {
    "slack.post_message": "Post a message to a Slack channel. You are always the author.",
    "slack.read_channel": "Read the messages in a Slack channel.",
    "slack.read_thread": "Read a Slack thread, including any approvals recorded in it.",
    "slack.search": "Search Slack messages by text.",
    "slack.request_approval": "Post a structured approval request to a channel.",
    "slack.get_user": "Look up a Slack user, including the roles they hold.",
    "tickets.get": "Fetch a support ticket by id.",
    "tickets.search": "Search support tickets by text.",
    "tickets.add_comment": "Add a comment to a ticket. The customer can read ticket comments.",
    "tickets.set_status": "Set a ticket's status.",
    "tickets.assign": "Assign a ticket to a user.",
    "billing.get_customer": "Fetch a customer's billing profile.",
    "billing.search_customers": "Search customers by name, email or external reference.",
    "billing.list_charges": "List a customer's charges.",
    "billing.get_charge": "Fetch a single charge.",
    "billing.list_refunds": "List the refunds recorded against a charge.",
    "billing.issue_refund": "Issue a refund against a charge.",
    "billing.update_customer": "Set a metadata key on a customer.",
    "billing.get_dispute": "Look up disputes by dispute id or by customer.",
    "billing.merge_customers": "Merge one customer profile into another. Irreversible.",
    "billing.delete_customer": "Delete a customer profile. Irreversible.",
    "policy.get_rule": "Retrieve a company policy rule by key.",
    "policy.search": "Search company policy rules by text.",
    "finish": "Declare the task complete, having taken the action it required.",
    "escalate": "Declare that this needs a human decision and stop.",
    "abstain": "Declare that no action should be taken, and stop.",
}

#: Bumped whenever the agent-visible schema changes shape.
#:
#: **1.0.0 -> 1.1.0 (2026-09-06).** ``billing.issue_refund.reason`` now
#: advertises its closed vocabulary. Previously the schema said ``string`` while
#: the handler accepted three values, so a schema-valid request could be
#: rejected by an internal conversion.
#:
#: A run's transcript cache is keyed on the request, which includes these
#: schemas, so **transcripts recorded under an earlier version cannot be
#: regenerated after a bump**. Runs record the version they used, and
#: regeneration reports a version mismatch rather than a bare cache miss.
TOOL_SCHEMA_VERSION = "1.1.0"

_JSON_TYPES = {"integer": "integer", "number": "number", "boolean": "boolean"}


def _resolve(field_schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Follow a ``$ref`` into ``$defs``.

    Pydantic emits enum fields as a reference to a definition holding the
    permitted values. Without following it the generated schema says only
    ``{"type": "string"}`` -- which is how the agent-visible contract came to
    advertise free text for a field the tool accepts three values for.
    """
    ref = field_schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return dict(defs.get(ref.split("/")[-1], {}))
    return field_schema


def _property_schema(
    field_schema: dict[str, Any], defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a Pydantic field schema into something a tool schema can carry."""
    defs = defs or {}
    field_schema = _resolve(field_schema, defs)
    if field_schema.get("allOf"):
        field_schema = _resolve(dict(field_schema["allOf"][0]), defs)
    if "anyOf" in field_schema:
        options = [o for o in field_schema["anyOf"] if o.get("type") != "null"]
        if options:
            field_schema = _resolve(dict(options[0]), defs)
    kind = str(field_schema.get("type") or "string")
    if kind == "array":
        return {"type": "array", "items": {"type": "string"}}
    schema: dict[str, Any] = {"type": _JSON_TYPES.get(kind, "string")}
    # A closed vocabulary is part of the contract, so the agent must be able to
    # see it. Omitting it makes a valid-looking request fail for reasons the
    # schema never disclosed.
    values = field_schema.get("enum")
    if isinstance(values, list) and values:
        schema["enum"] = [str(v) for v in values]
    return schema


def schema_for(action_type: type[BaseAction]) -> dict[str, Any]:
    """A tool schema for one action model."""
    model_schema = action_type.model_json_schema()
    defs = dict(model_schema.get("$defs", {}))
    kind = str(action_type.model_fields["kind"].default)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, field in action_type.model_fields.items():
        if name == "kind":
            continue
        raw = model_schema.get("properties", {}).get(name, {})
        properties[name] = _property_schema(raw, defs)
        if field.is_required():
            required.append(name)
    return {
        "name": kind.replace(".", "__"),
        "description": DESCRIPTIONS.get(kind, kind),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def all_tool_schemas() -> list[dict[str, Any]]:
    """Every action an agent may take. Excludes the malformed-action sentinel."""
    return [
        schema_for(action_type)
        for action_type in ACTION_TYPES
        if action_type is not MalformedAction
    ]


def tool_name_to_kind(name: str) -> str:
    return name.replace("__", ".")


ACTION_KINDS: frozenset[str] = frozenset(
    k.value for k in ActionKind if k is not ActionKind.MALFORMED
)


def tool_schema_hash() -> str:
    """Content hash of the agent-visible schemas.

    Recorded per run: two runs that disagree on this were shown different tools,
    which is a different experiment even if everything else matches.
    """
    from cerl.core import content_hash

    return content_hash(all_tool_schemas())
