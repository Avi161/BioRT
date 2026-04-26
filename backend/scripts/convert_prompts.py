"""Convert prompt CSV into rich prompt JSON schema (v1).

Usage:
    python scripts/convert_prompts.py --input docs/prompts.csv --output prompts/real_prompts.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

CANONICAL_CATEGORIES = (
    "synthesis_evasion",
    "lab_automation_uplift",
    "dual_use_protocol",
    "acquisition_kyc",
    "virology",
)

REQUIRED_COLUMNS = (
    "category",
    "prompt_id",
    "prompt_text",
    "abstraction_check",
)


def _normalize(value: str | None) -> str:
    return (value or "").strip()


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [f.strip() for f in (reader.fieldnames or [])]
        missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. Found columns: {fieldnames}"
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({k: _normalize(v) for k, v in row.items()})
    return rows


def convert_csv_to_schema(
    input_path: Path,
    output_path: Path,
    source: str,
    min_per_category: int = 5,
) -> dict:
    rows = _load_rows(input_path)

    filtered = [r for r in rows if _normalize(r.get("abstraction_check")) == "Pass"]
    if not filtered:
        raise ValueError("No rows with abstraction_check == 'Pass' were found.")

    prompt_ids = [_normalize(r.get("prompt_id")) for r in filtered]
    missing_prompt_ids = [pid for pid in prompt_ids if not pid]
    if missing_prompt_ids:
        raise ValueError("At least one retained row is missing prompt_id.")

    dupes = [pid for pid, count in Counter(prompt_ids).items() if count > 1]
    if dupes:
        raise ValueError(f"Duplicate prompt_id values found: {sorted(dupes)}")

    categories: dict[str, list[dict[str, str]]] = {c: [] for c in CANONICAL_CATEGORIES}
    for row in filtered:
        category = _normalize(row.get("category"))
        if category not in categories:
            raise ValueError(
                f"Invalid category {category!r}. Must be one of: {list(CANONICAL_CATEGORIES)}"
            )

        prompt_text = _normalize(row.get("prompt_text"))
        if not prompt_text:
            raise ValueError(f"Row with prompt_id={row.get('prompt_id')!r} missing prompt_text")

        obj = {
            "prompt_id": _normalize(row.get("prompt_id")),
            "prompt_text": prompt_text,
            "abstraction_check": "Pass",
        }
        if _normalize(row.get("source")):
            obj["source"] = _normalize(row.get("source"))
        if _normalize(row.get("notes")):
            obj["notes"] = _normalize(row.get("notes"))
        categories[category].append(obj)

    too_small = {cat: len(items) for cat, items in categories.items() if len(items) < min_per_category}
    if too_small:
        raise ValueError(
            "Refusing to emit JSON. Category counts below minimum "
            f"{min_per_category}: {too_small}"
        )

    doc = {
        "schema_version": 1,
        "source": source,
        "abstraction_policy": "Only entries with abstraction_check == \"Pass\" are included.",
        "categories": categories,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert prompt CSV to rich prompt JSON.")
    parser.add_argument("--input", required=True, help="Path to source CSV")
    parser.add_argument(
        "--output",
        default="prompts/real_prompts.json",
        help="Destination JSON path (default: prompts/real_prompts.json)",
    )
    parser.add_argument(
        "--source",
        default="C1 spreadsheet export",
        help="Provenance string saved in output JSON",
    )
    parser.add_argument(
        "--min-per-category",
        type=int,
        default=5,
        help="Minimum retained prompt rows required per canonical category",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    doc = convert_csv_to_schema(
        input_path=input_path,
        output_path=output_path,
        source=args.source,
        min_per_category=args.min_per_category,
    )

    counts = {
        category: len(prompts)
        for category, prompts in doc["categories"].items()
    }
    print(
        " | ".join(f"{category}: {count}" for category, count in counts.items())
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
