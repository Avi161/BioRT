"""Tests for scripts/convert_prompts.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "convert_prompts.py"
CANONICAL_CATEGORIES = [
    "synthesis_evasion",
    "lab_automation_uplift",
    "dual_use_protocol",
    "acquisition_kyc",
    "virology",
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["category", "prompt_id", "prompt_text", "abstraction_check", "source", "notes"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_convert_prompts_cli_happy_path(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    output_json = tmp_path / "real_prompts.json"

    rows: list[dict[str, str]] = []
    for idx, category in enumerate(CANONICAL_CATEGORIES, start=1):
        rows.append(
            {
                "category": category,
                "prompt_id": f"{category}-ok",
                "prompt_text": f"Safe abstract prompt {idx}",
                "abstraction_check": "Pass",
                "source": "unit-test",
                "notes": "",
            }
        )
        rows.append(
            {
                "category": category,
                "prompt_id": f"{category}-drop",
                "prompt_text": "This should be dropped",
                "abstraction_check": "Fail",
                "source": "unit-test",
                "notes": "",
            }
        )

    _write_csv(input_csv, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_csv),
            "--output",
            str(output_json),
            "--source",
            "unit-test-export",
            "--min-per-category",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source"] == "unit-test-export"
    assert set(payload["categories"].keys()) == set(CANONICAL_CATEGORIES)

    for category in CANONICAL_CATEGORIES:
        prompts = payload["categories"][category]
        assert len(prompts) == 1
        assert prompts[0]["abstraction_check"] == "Pass"
        assert prompts[0]["prompt_id"].endswith("-ok")

    assert "synthesis_evasion: 1" in result.stdout


def test_convert_prompts_cli_fails_on_missing_required_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "bad.csv"
    output_json = tmp_path / "real_prompts.json"

    # Intentionally omit required 'prompt_text' and 'abstraction_check'
    with input_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "prompt_id"])
        writer.writeheader()
        writer.writerow({"category": "synthesis_evasion", "prompt_id": "missing-fields"})

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_csv),
            "--output",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing required columns" in result.stderr.lower() or "missing required columns" in result.stdout.lower()
