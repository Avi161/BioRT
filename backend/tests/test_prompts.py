"""Tests for prompt datasets — mock (legacy) and real (BioRT-Bench)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

MOCK_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "mock_prompts.json"
REAL_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts_long.json"

EXPECTED_CATEGORIES = {
    "synthesis_evasion",
    "lab_automation_uplift",
    "dual_use_protocol",
    "acquisition_kyc",
    "virology",
}


# ---------------------------------------------------------------------------
# Legacy mock prompts (flat schema) — skip if file was removed
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_prompts_data() -> dict[str, list[str]]:
    with open(MOCK_PROMPTS_PATH) as fh:
        return json.load(fh)


_mock_missing = not MOCK_PROMPTS_PATH.exists()
_mock_skip = pytest.mark.skipif(_mock_missing, reason="mock_prompts.json not present")


@_mock_skip
class TestMockPrompts:
    def test_file_exists(self) -> None:
        assert MOCK_PROMPTS_PATH.exists()

    def test_has_all_categories(self, mock_prompts_data: dict) -> None:
        assert set(mock_prompts_data.keys()) == EXPECTED_CATEGORIES

    def test_two_prompts_per_category(self, mock_prompts_data: dict) -> None:
        for category, prompts in mock_prompts_data.items():
            assert len(prompts) == 2, f"{category} should have 2 prompts"

    def test_prompts_are_nonempty_strings(self, mock_prompts_data: dict) -> None:
        for category, prompts in mock_prompts_data.items():
            for prompt in prompts:
                assert isinstance(prompt, str) and len(prompt) > 10, (
                    f"Invalid prompt in {category}"
                )


# ---------------------------------------------------------------------------
# Real BioRT-Bench prompts (rich schema v1)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_prompts_doc() -> dict:
    with open(REAL_PROMPTS_PATH) as fh:
        return json.load(fh)


class TestRealPrompts:
    def test_file_exists(self) -> None:
        assert REAL_PROMPTS_PATH.exists()

    def test_schema_version(self, real_prompts_doc: dict) -> None:
        assert real_prompts_doc.get("schema_version") == 1

    def test_has_all_categories(self, real_prompts_doc: dict) -> None:
        assert set(real_prompts_doc["categories"].keys()) == EXPECTED_CATEGORIES

    def test_eight_prompts_per_category(self, real_prompts_doc: dict) -> None:
        for category, prompts in real_prompts_doc["categories"].items():
            assert len(prompts) == 8, (
                f"{category} should have 8 prompts, got {len(prompts)}"
            )

    def test_prompt_objects_have_required_fields(self, real_prompts_doc: dict) -> None:
        for category, prompts in real_prompts_doc["categories"].items():
            for obj in prompts:
                assert "prompt_id" in obj, f"Missing prompt_id in {category}"
                assert "prompt_text" in obj, f"Missing prompt_text in {category}"
                assert "abstraction_check" in obj, (
                    f"Missing abstraction_check in {category}"
                )
                assert isinstance(obj["prompt_id"], str), (
                    f"prompt_id must be str in {category}, got {type(obj['prompt_id']).__name__}"
                )
                assert isinstance(obj["prompt_text"], str), (
                    f"prompt_text must be str in {category}, got {type(obj['prompt_text']).__name__}"
                )

    def test_all_abstraction_checks_pass(self, real_prompts_doc: dict) -> None:
        for category, prompts in real_prompts_doc["categories"].items():
            for obj in prompts:
                assert obj["abstraction_check"] == "Pass", (
                    f"{obj['prompt_id']} in {category} has "
                    f"abstraction_check={obj['abstraction_check']!r}, expected 'Pass'"
                )

    def test_prompt_ids_are_unique(self, real_prompts_doc: dict) -> None:
        all_ids: list[str] = []
        for prompts in real_prompts_doc["categories"].values():
            all_ids.extend(obj["prompt_id"] for obj in prompts)
        counts = Counter(all_ids)
        duplicates = sorted([prompt_id for prompt_id, count in counts.items() if count > 1])
        assert not duplicates, f"Duplicate prompt_ids found: {duplicates}"

    def test_prompt_texts_are_nonempty(self, real_prompts_doc: dict) -> None:
        for category, prompts in real_prompts_doc["categories"].items():
            for obj in prompts:
                assert isinstance(obj["prompt_text"], str) and len(obj["prompt_text"]) > 10, (
                    f"Invalid prompt_text for {obj['prompt_id']} in {category}"
                )
