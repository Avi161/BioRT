"""CLI orchestrator: walk attack-run JSONs and write scored eval JSONs.

Reads every ``backend/results/{provider}/{method}/*.json`` produced by
``validate_attacks.py`` / ``matrix_runner.py``, runs the bio-aware judge in
``backend/judge.py``, and writes one eval JSON per source to
``backend/eval_results/{provider}/{method}/<same filename>.json``.

Usage:
    python score_results.py                       # score everything new
    python score_results.py --force               # rescore even existing
    python score_results.py --limit 2             # cap files (debug)
    python score_results.py --judge-provider deepseek

Exit codes:
    0 — every file scored or skipped cleanly
    1 — at least one file errored mid-pipeline
    2 — config error (no rubric, missing key, bad provider)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("score_results")

from pyrit.setup import IN_MEMORY, initialize_pyrit_async  # noqa: E402

import judge  # noqa: E402
from config.models import MODEL_REGISTRY, ModelConfig, build_target  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_EVAL_DIR = Path(__file__).parent / "eval_results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score attack-run JSONs with the bio-aware LLM judge."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score even when an eval JSON already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many files (debug).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Input directory of attack-run JSONs.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Output directory for eval JSONs.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Override JUDGE_PROVIDER env (e.g. 'moonshot', 'deepseek').",
    )
    return parser.parse_args(argv)


def _resolve_judge_cfg(provider: str) -> ModelConfig:
    cfg = next((c for c in MODEL_REGISTRY if c.provider == provider), None)
    if cfg is None:
        known = sorted(c.provider for c in MODEL_REGISTRY)
        raise EnvironmentError(
            f"Unknown judge provider {provider!r}. Known: {known}."
        )
    return cfg


async def _score_one(
    src_path: Path,
    out_path: Path,
    judge_target: Any,
    judge_cfg: ModelConfig,
) -> tuple[bool, dict[str, Any]]:
    attack_run = json.loads(src_path.read_text())
    eval_run = await judge.score_attack_run(attack_run, judge_target, judge_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"attack_run": attack_run, "eval_run": eval_run},
            indent=2,
            default=str,
        )
    )
    return True, eval_run


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Fail loudly if rubric not pasted (also stamped into every eval record).
    try:
        version = judge.prompt_version()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Rubric version: %s", version)

    provider = (
        args.judge_provider
        or os.getenv("JUDGE_PROVIDER")
        or judge.DEFAULT_JUDGE_PROVIDER
    ).strip()
    try:
        judge_cfg = _resolve_judge_cfg(provider)
    except EnvironmentError as exc:
        logger.error("%s", exc)
        return 2

    # Override max_completion_tokens for the judge only; registry row stays
    # untouched so attack-pipeline behaviour is unchanged.
    judge_cfg = dataclasses.replace(
        judge_cfg, max_completion_tokens=judge.JUDGE_MAX_COMPLETION_TOKENS
    )

    await initialize_pyrit_async(memory_db_type=IN_MEMORY)
    try:
        judge_target = build_target(judge_cfg)
    except EnvironmentError as exc:
        logger.error("Cannot build judge target: %s", exc)
        return 2
    logger.info(
        "Judge LLM: %s (%s) max_completion_tokens=%d",
        judge_cfg.display_name,
        judge_cfg.provider,
        judge_cfg.max_completion_tokens,
    )

    results_dir: Path = args.results_dir
    eval_dir: Path = args.eval_dir
    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        return 2

    paths = sorted(results_dir.rglob("*.json"))
    if not paths:
        logger.warning("No attack-run JSONs found under %s", results_dir)
        return 0

    n_scored = 0
    n_skipped_exists = 0
    n_errored = 0
    for src in paths:
        if args.limit is not None and n_scored >= args.limit:
            logger.info("Hit --limit %d; stopping.", args.limit)
            break

        rel = src.relative_to(results_dir)
        out = eval_dir / rel
        if out.exists() and not args.force:
            logger.info("skip (exists): %s", rel)
            n_skipped_exists += 1
            continue

        try:
            _, eval_run = await _score_one(src, out, judge_target, judge_cfg)
        except Exception as exc:  # noqa: BLE001
            logger.exception("scoring failed for %s: %r", rel, exc)
            n_errored += 1
            continue

        status = eval_run.get("parse_status")
        asr = eval_run.get("scores", {}).get("ASR")
        if status == "ok" and asr is not None:
            logger.info("scored: %s [ASR=%.3f]", rel, asr)
        elif status == "skipped":
            logger.info(
                "scored (stub: %s): %s", eval_run.get("skip_reason"), rel
            )
        else:
            logger.warning("scored (parse %s): %s", status, rel)
        n_scored += 1

    logger.info(
        "summary: scored=%d skipped_existing=%d errored=%d total_seen=%d",
        n_scored,
        n_skipped_exists,
        n_errored,
        len(paths),
    )
    return 0 if n_errored == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
