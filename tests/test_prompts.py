"""Tests for mock_prompts.json — validate structure and content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "mock_prompts.json"
EXPECTED_CATEGORIES = {
    "synthesis_evasion",
    "lab_automation_uplift",
    "dual_use_protocol",
    "acquisition_kyc",
    "virology",
}


@pytest.fixture
def prompts_data() -> dict[str, list[str]]:
    with open(PROMPTS_PATH) as fh:
        return json.load(fh)


class TestMockPrompts:
    def test_file_exists(self) -> None:
        assert PROMPTS_PATH.exists()

    def test_has_all_categories(self, prompts_data: dict) -> None:
        assert set(prompts_data.keys()) == EXPECTED_CATEGORIES

    def test_two_prompts_per_category(self, prompts_data: dict) -> None:
        for category, prompts in prompts_data.items():
            assert len(prompts) == 2, f"{category} should have 2 prompts"

    def test_prompts_are_nonempty_strings(self, prompts_data: dict) -> None:
        for category, prompts in prompts_data.items():
            for prompt in prompts:
                assert isinstance(prompt, str) and len(prompt) > 10, (
                    f"Invalid prompt in {category}"
                )
