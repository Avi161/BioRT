"""Tests for Crescendo debug helper module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from backend import crescendo_debug


def test_default_debug_output_dir_is_single_repo_path() -> None:
    """Crescendo JSONL is written under the repo, independent of process cwd."""
    p = Path(crescendo_debug.DEFAULT_DEBUG_OUTPUT_DIR)
    assert p.is_absolute()
    assert p.name == "crescendo" and p.parent.name == "results"


def test_load_completed_prompt_ids_from_jsonl() -> None:
    """Resume helper collects prompt_id from each line's metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "partial.jsonl"
        line1 = {
            "metadata": {"prompt_id": "P1", "category": "synthesis_evasion"},
        }
        line2 = {
            "metadata": {"prompt_id": "P2", "category": "synthesis_evasion"},
        }
        p.write_text(
            json.dumps(line1) + "\n" + json.dumps(line2) + "\n",
            encoding="utf-8",
        )
        got = crescendo_debug.load_completed_prompt_ids_from_jsonl(str(p))
        assert got == {"P1", "P2"}


def test_run_crescendo_moonshot_as_both_roles_is_exported() -> None:
    assert callable(crescendo_debug.run_crescendo_moonshot_as_both_roles)
    assert crescendo_debug.run_crescendo_moonshot_as_both_roles.__doc__


def test_run_crescendo_provider_modes_exported() -> None:
    """Exported provider entrypoints include Kimi->Claude plus legacy modes."""
    assert callable(crescendo_debug.run_crescendo_anthropic_as_both_roles)
    assert callable(crescendo_debug.run_crescendo_openai_as_both_roles)
    assert callable(crescendo_debug.run_crescendo_kimi_attacks_anthropic)
    assert crescendo_debug.DEFAULT_DEBUG_ATTACKER_PROVIDER == "moonshot"
    assert "moonshot" in crescendo_debug.run_crescendo_anthropic_as_both_roles.__doc__.lower()
    assert "moonshot" in crescendo_debug.run_crescendo_openai_as_both_roles.__doc__.lower()
    assert "anthropic" in crescendo_debug.run_crescendo_kimi_attacks_anthropic.__doc__.lower()


def test_default_debug_prompt_files_are_expected() -> None:
    assert crescendo_debug.DEFAULT_DEBUG_PROMPT_FILES == (
        "prompts/prompts_long.json",
        "backend/prompts/prompts_short.json",
    )


def test_default_short_bench_path_matches_long_count() -> None:
    """Short and long benchmark files should list the same number of prompts."""
    assert crescendo_debug.DEFAULT_SHORT_BENCH_PROMPT_FILE.endswith("prompts_short.json")
    short_cases = crescendo_debug.load_all_prompt_cases(
        crescendo_debug.DEFAULT_SHORT_BENCH_PROMPT_FILE
    )
    long_cases = crescendo_debug.load_all_prompt_cases("prompts/prompts_long.json")
    assert len(short_cases) == len(long_cases) == 40


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
    assert crescendo_debug._is_defender_refusal_prose("I can't help with that request.") is True
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


def test_defender_content_filter_counts_as_refusal() -> None:
    """API policy blocks (empty content + content_filter) are counted for Anthropic etc."""
    blob = (
        '{"choices":[{"finish_reason":"content_filter",'
        '"message":{"content":""}}]}'
    )
    # Structured payload (and data_type error) both classify as refusal for metrics.
    assert crescendo_debug._defender_turn_counts_as_refusal(blob, "text") is True
    assert crescendo_debug._defender_turn_counts_as_refusal(blob, "error") is True
    transcript = [
        {
            "role": "user",
            "original_value": "q1",
            "converted_value": "q1",
            "data_type": "text",
        },
        {
            "role": "assistant",
            "original_value": blob,
            "converted_value": blob,
            "data_type": "error",
        },
    ]
    _, _, summary = crescendo_debug._split_conversations(transcript)
    assert summary["defender_refusals"] == 1
