#!/usr/bin/env python3
"""Count .jsonl entries by status, grouped by model -> method -> source_line_index."""

import json
from collections import defaultdict
from pathlib import Path

SEARCH_DIR = Path("../eval_results")
STATUSES = ("ok", "filter", "judge_refused", "skipped", "error", "empty_response", "transient_error")


def parse_path(jsonl_path: Path):
    """Extract (model, method, source_line_index) from path relative to SEARCH_DIR.

    Adjust this if your directory layout differs. Default assumes:
        ../eval_results/<model>/<method>/<source_line_index>.jsonl
    Falls back to using parent dirs / filename as best-effort.
    """
    rel = jsonl_path.relative_to(SEARCH_DIR.resolve())
    parts = rel.parts
    if len(parts) >= 3:
        model = parts[0]
        method = parts[1]
        source_line_index = jsonl_path.stem
    elif len(parts) == 2:
        model, method = parts[0], "unknown"
        source_line_index = jsonl_path.stem
    else:
        model, method = "unknown", "unknown"
        source_line_index = jsonl_path.stem
    return model, method, source_line_index


def count_file(jsonl_path: Path):
    counts = {s: 0 for s in STATUSES}
    counts["other"] = 0
    asr_buckets = {"zero": 0, "low": 0, "high": 0}  # =0, (0, 0.5], (0.5, 1]
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counts["other"] += 1
                continue
            eval_run = obj.get("eval_run") or {}
            status = eval_run.get("status", "other") if isinstance(eval_run, dict) else "other"
            if status in counts:
                counts[status] += 1
            else:
                counts["other"] += 1

            score = eval_run.get("scores") if isinstance(eval_run, dict) else None
            if isinstance(score, dict) and "ASR" in score:
                asr = score["ASR"]
                if isinstance(asr, (int, float)):
                    if asr == 0:
                        asr_buckets["zero"] += 1
                    elif 0 < asr <= 0.5:
                        asr_buckets["low"] += 1
                    elif 0.5 < asr <= 1:
                        asr_buckets["high"] += 1
    return counts, asr_buckets


def main():
    if not SEARCH_DIR.exists():
        print(f"Directory not found: {SEARCH_DIR.resolve()}")
        return

    base = SEARCH_DIR.resolve()
    files = sorted(base.rglob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {base}")
        return

    # nested: model -> method -> source_line_index -> (counts, asr_buckets)
    tree = defaultdict(lambda: defaultdict(dict))
    for f in files:
        model, method, sli = parse_path(f)
        tree[model][method][sli] = count_file(f)

    # column width per status — wide enough for "transient_error=9999"
    status_w = max(len(s) for s in STATUSES) + 6  # "<status>=NNNN  "
    asr_labels = [("=0   ", "zero"), ("(0,.5]", "low"), ("(.5,1]", "high")]

    for model in sorted(tree):
        print(f"\n╔══ {model} {'═' * max(0, 60 - len(model))}")
        for method in sorted(tree[model]):
            method_totals = defaultdict(int)
            method_asr = {"zero": 0, "low": 0, "high": 0}
            for sli in sorted(tree[model][method]):
                counts, asr = tree[model][method][sli]
                for k, v in counts.items():
                    method_totals[k] += v
                for k, v in asr.items():
                    method_asr[k] += v

            status_cells = []
            for s in STATUSES:
                if method_totals[s] > 0:
                    status_cells.append(f"{s}={method_totals[s]}".ljust(status_w))
            if method_totals["other"] > 0:
                status_cells.append(f"other={method_totals['other']}".ljust(status_w))
            status_str = "".join(status_cells) if status_cells else "(empty)"

            asr_str = "  ".join(f"asr{label}={method_asr[key]}" for label, key in asr_labels)

            print(f"  ▸ {method}")
            print(f"      {status_str}")
            print(f"      {asr_str}")


if __name__ == "__main__":
    main()