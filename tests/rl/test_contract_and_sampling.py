"""R2 (tool contract), R3 (sampling), R4 (parsing).

Tokenizer-only and pure-Python; nothing here loads model weights.
"""

from __future__ import annotations

import pytest

from cerl.agents.tool_schemas import all_tool_schemas
from cerl_rl.environment import PUBLIC_TOOLS, RESERVED_ARGUMENTS, parse_action
from cerl_rl.tools import tool_contract, tool_menu_v1

# --------------------------------------------------------------------------
# R2: the policy must see the whole advertised contract
# --------------------------------------------------------------------------


def test_every_public_tool_appears_with_its_description():
    contract = tool_contract()
    for schema in all_tool_schemas():
        assert schema["name"] in contract
        description = schema.get("description", "").strip()
        if description:
            assert description in contract, schema["name"]


def test_enumerated_values_are_shown_rather_than_left_to_be_guessed():
    """The concrete failure: four recorded malformed attempts set a ticket to
    ``closed``, a value the schema excludes and the old menu never showed."""
    contract = tool_contract()
    assert "open|pending_customer|resolved|escalated" in contract
    assert "closed" not in contract
    # the old rendering showed none of it
    assert "open|pending_customer" not in tool_menu_v1()


def test_numeric_bounds_and_requiredness_are_shown():
    contract = tool_contract()
    assert "amount_cents: integer, > 0, required" in contract
    assert "optional" in contract          # non-required arguments are marked


def test_closed_argument_sets_are_stated():
    assert "no other arguments are accepted" in tool_contract()


def test_the_contract_leaks_no_privileged_material(all_frozen):
    """It describes what can be asked for, never what is correct."""
    contract = tool_contract()
    for scenario in [s for s in all_frozen if s.family == "duplicate_charge_approval"][:20]:
        assert scenario.branch not in contract
        for item in scenario.rubric:
            assert item.id not in contract


def test_the_complete_contract_is_substantially_larger_than_the_old_menu():
    """Recorded so the prompt-budget recomputation has a stated basis."""
    assert len(tool_contract()) > 4 * len(tool_menu_v1())


# --------------------------------------------------------------------------
# R4: arguments must not be able to choose the tool
# --------------------------------------------------------------------------


def test_a_kind_argument_cannot_override_the_selected_tool():
    """The review's reproduction: this parsed as BillingDeleteCustomer."""
    action = parse_action(
        '{"tool":"tickets__get","arguments":'
        '{"kind":"billing.delete_customer","customer_id":"cus_000000000001"}}',
    )
    assert str(action.kind) == "malformed"
    assert "kind" in action.parse_error


def test_kind_is_the_reserved_name_and_the_check_is_not_a_special_case():
    assert "kind" in RESERVED_ARGUMENTS


@pytest.mark.parametrize("arguments", ["0", "[]", '""', "3.5", '"text"'])
def test_a_non_object_arguments_value_is_refused_not_coerced(arguments):
    """``payload.get("arguments") or {}`` silently invented an empty call."""
    action = parse_action(f'{{"tool":"policy__search","arguments":{arguments}}}')
    assert str(action.kind) == "malformed"
    assert "must be an object" in action.parse_error


def test_a_missing_or_null_arguments_value_is_treated_as_no_arguments():
    """Distinct from the case above: absent is not the same as wrong-typed."""
    for text in ('{"tool":"policy__search"}', '{"tool":"policy__search","arguments":null}'):
        action = parse_action(text)
        # policy.search requires a query, so this is malformed for the *right*
        # reason -- a missing required argument, not a type error.
        assert str(action.kind) == "malformed"
        assert "must be an object" not in action.parse_error


def test_the_tool_name_is_looked_up_in_the_public_allowlist():
    """``tool_name_to_kind`` is a string substitution and will transform a name
    no tool has; the allowlist is what makes an unknown tool malformed."""
    assert len(PUBLIC_TOOLS) == len(all_tool_schemas())
    action = parse_action('{"tool":"billing__wire_money","arguments":{}}')
    assert str(action.kind) == "malformed"
    assert "not in the advertised tool list" in action.parse_error


def test_a_legitimate_call_still_parses():
    action = parse_action(
        '{"tool":"tickets__get","arguments":{"ticket_id":"tkt_000000000001"}}',
    )
    assert str(action.kind) == "tickets.get"


def test_every_advertised_tool_name_maps_to_a_real_action_kind():
    for name, kind in PUBLIC_TOOLS.items():
        assert kind and "__" not in kind, name


# --------------------------------------------------------------------------
# R3: sampling and scoring must be the same distribution
# --------------------------------------------------------------------------


def test_v2_neutralises_every_truncating_sampler():
    """min_p is neutralised by *omission* -- see the test below for why."""
    from cerl_rl import protocol_v2

    neutral = protocol_v2.SAMPLING_NEUTRALISED
    assert neutral["top_k"] == 0
    assert neutral["top_p"] == 1.0
    assert neutral["typical_p"] == 1.0
    assert neutral["repetition_penalty"] == 1.0
    assert "min_p" not in neutral


@pytest.mark.model
def test_the_models_own_defaults_would_otherwise_truncate():
    """Why the neutralisation is needed at all: the pinned config sets top_k=20,
    which v1 never overrode while differentiating the untruncated softmax."""
    from transformers import GenerationConfig

    from cerl_rl import protocol

    config = GenerationConfig.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION,
    )
    assert config.top_k == 20
    assert config.top_p == 0.95


@pytest.mark.model
def test_the_v2_sampling_arguments_override_those_defaults():
    from transformers import GenerationConfig

    from cerl_rl import protocol, protocol_v2

    config = GenerationConfig.from_pretrained(
        protocol.MODEL_ID, revision=protocol.MODEL_REVISION,
    )
    merged = config.to_dict() | dict(protocol_v2.SAMPLING_NEUTRALISED)
    assert merged["top_k"] == 0
    assert merged["top_p"] == 1.0


# --------------------------------------------------------------------------
# the min_p trap that aborted the first v2 trial
# --------------------------------------------------------------------------


def test_min_p_is_omitted_rather_than_set_to_zero():
    """Setting it to 0.0 does not disable it -- it builds the warper.

    ``transformers`` guards min-p on a bare ``is not None``, and ``0.0 is not
    None``. The constructed warper then sorts, gathers and scatters across the
    whole vocabulary, which aborted the process on MPS and killed the first v2
    training trial. Omitting the key leaves it ``None`` and builds nothing,
    with identical intended semantics.
    """
    from cerl_rl import protocol_v2

    assert "min_p" not in protocol_v2.SAMPLING_NEUTRALISED


@pytest.mark.model
def test_every_neutralised_setting_actually_disables_its_warper():
    """Checked against the installed source, not assumed.

    A neutral *value* only disables a warper if the guard tests the value.
    ``min_p`` was the one that did not, and nothing stops another key from
    joining it in a future release.
    """
    import inspect

    from transformers.generation.utils import GenerationMixin

    from cerl_rl import protocol_v2

    source = inspect.getsource(GenerationMixin._get_logits_processor)
    guards = {
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("if") and "generation_config." in line
    }
    for key in protocol_v2.SAMPLING_NEUTRALISED:
        guard = next((g for g in guards if f"generation_config.{key}" in g), None)
        assert guard is not None, f"no guard found for {key}"
        tests_value = any(token in guard for token in ("!=", "<", ">", "is True"))
        assert tests_value, (
            f"{key} is guarded only by a None check, so its neutral value would "
            f"still construct the warper -- omit the key instead of neutralising it"
        )
