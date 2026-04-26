"""Tests for Crescendo debug helper module."""

from __future__ import annotations

from types import SimpleNamespace

from backend import crescendo_debug


def test_default_debug_prompt_files_are_expected() -> None:
    assert crescendo_debug.DEFAULT_DEBUG_PROMPT_FILES == (
        "prompts/prompts_long.json",
        "backend/prompts/prompts_short.json",
    )


def test_load_all_prompt_cases_bior_bench_is_40() -> None:
    cases = crescendo_debug.load_all_prompt_cases("prompts/prompts_long.json")
    assert len(cases) == 40
    ids = [c.prompt_id for c in cases if c.prompt_id]
    assert len(ids) == 40
    categories = {c.category for c in cases}
    assert categories == {
        "synthesis_evasion",
        "lab_automation_uplift",
        "dual_use_protocol",
        "acquisition_kyc",
        "virology",
    }


def test_load_first_prompt_case_for_long_prompts() -> None:
    case = crescendo_debug.load_first_prompt_case("prompts/prompts_long.json")
    assert case.category == "synthesis_evasion"
    assert case.prompt_id == "SCREEN-01"
    assert case.prompt_file.endswith("backend/prompts/prompts_long.json")
    assert "orders keep getting rejected" in case.prompt


def test_load_first_prompt_case_for_short_prompts() -> None:
    case = crescendo_debug.load_first_prompt_case("backend/prompts/prompts_short.json")
    assert case.category == "synthesis_evasion"
    assert case.prompt_id == "SCREEN-01"
    assert case.prompt_file.endswith("backend/prompts/prompts_short.json")
    assert "orders keep getting rejected" in case.prompt


def test_select_model_config_raises_when_provider_missing() -> None:
    configs = [
        SimpleNamespace(
            display_name="DeepSeek V4 Flash",
            provider="deepseek",
            endpoint="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            model_name="deepseek-chat",
        ),
    ]
    try:
        crescendo_debug.select_model_config(configs, "moonshot")
    except RuntimeError as exc:
        assert "Provider 'moonshot' is not available" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing provider")


def test_split_conversations_separates_attacker_and_defender_turns() -> None:
    transcript = [
        {
            "role": "user",
            "original_value": "attacker q1",
            "converted_value": "attacker q1",
            "data_type": "text",
        },
        {
            "role": "assistant",
            "original_value": "defender a1",
            "converted_value": "defender a1",
            "data_type": "text",
        },
        {
            "role": "user",
            "original_value": "attacker q2",
            "converted_value": "attacker q2",
            "data_type": "text",
        },
    ]

    attacker_conversation, defender_conversation, summary = (
        crescendo_debug._split_conversations(transcript)
    )

    assert len(attacker_conversation) == 2
    assert len(defender_conversation) == 1
    assert summary["kimi_prompt_count"] == 2
    assert summary["defender_reply_count"] == 1
    assert summary["defender_refusals"] == 0
    assert attacker_conversation[0]["turn_index"] == 1
    assert attacker_conversation[0]["crescendo_round"] == 1
    assert defender_conversation[0]["turn_index"] == 2
    assert defender_conversation[0]["crescendo_round"] == 1
    assert attacker_conversation[1]["turn_index"] == 3
    assert attacker_conversation[1]["crescendo_round"] == 2


def test_split_conversations_drops_scorer_user_messages() -> None:
    transcript = [
        {
            "role": "user",
            "original_value": "conversation_objective: x\n",
            "converted_value": "conversation_objective: x\n",
            "data_type": "text",
        },
        {
            "role": "user",
            "original_value": "Real Kimi line to target",
            "converted_value": "Real Kimi line to target",
            "data_type": "text",
        },
        {
            "role": "assistant",
            "original_value": "Helpful answer",
            "converted_value": "Helpful answer",
            "data_type": "text",
        },
    ]
    a_conv, d_conv, summary = crescendo_debug._split_conversations(transcript)
    assert summary["kimi_prompt_count"] == 1
    assert len(a_conv) == 1
    assert a_conv[0]["text"] == "Real Kimi line to target"
    assert len(d_conv) == 1


def test_defender_refusal_increments_summary() -> None:
    assert crescendo_debug._is_defender_refusal("I can't help with that request.") is True
    transcript = [
        {
            "role": "user",
            "original_value": "q1",
            "converted_value": "q1",
            "data_type": "text",
        },
        {
            "role": "assistant",
            "original_value": "Sorry, I can't help with that.",
            "converted_value": "Sorry, I can't help with that.",
            "data_type": "text",
        },
    ]
    _, _, summary = crescendo_debug._split_conversations(transcript)
    assert summary["defender_refusals"] == 1
