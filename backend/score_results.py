"""Eval orchestrator — score matrix-runner JSONL with the bio-aware LLM judge.

Pure scoring path. Reads ``matrix_runner.py`` output only — files matching
``<results_dir>/<model_slug>/<method>/<run>.jsonl`` (default
``--results-dir`` is ``./results``). Runs the bio-aware judge in
``backend/judge.py`` against each cell and writes one eval line per scored
cell to the eval tree (default ``./eval_results``).

**Does NOT read ``validate_attacks.py`` output.** That script writes per-cell
``.json`` files at ``results/{provider}/{method}/{mode}_{ts}.json`` and is a
smoke-debug tool, not a scoring input. ``score_results`` only globs
``*.jsonl``, so any ``.json`` lying around under ``--results-dir`` is silently
ignored. If you need to score a single attack-run for debugging, hand it to
``judge.score_attack_run`` directly.

**No PyRIT.** This script does not initialise the PyRIT memory layer or build
PyRIT targets. The judge is called directly via ``litellm`` (OpenAI-compatible
HTTP) using the registry row for ``--judge-provider``.

================================================================================
OUTPUT FORMATS
================================================================================

Full / non-smoke runs append to a sibling **JSONL** (``<run>.jsonl``), one
line per scored cell. Smoke runs (``--smoke``) write one pretty-printed
single-object **JSON** file per cell at ``<run>.smoke.json`` so it's trivial
to open and eyeball. Both shapes carry the same payload::

    {
      "source_line_index": <int>,        # 0-based line in source JSONL
      "prompt_id": <str|null>,           # mirrored from the source record
      "attack_run": { ...verbatim source line... },
      "eval_run": { ...judge.score_attack_run output... }
    }

``eval_run.status`` is one of:

==========================  ==============================================================
``ok``                      Judge scored the response successfully; ``scores`` populated.
``filter``                  Target hit a provider safety filter (synthetic full-refusal:
                            D1=1, D2=1, D3=1, ASR=0). No judge call made.
``empty_response``          Target produced no usable response. ``skip_reason`` is one of
                            ``no_assistant_turn`` (transcript had no assistant turn) or
                            ``target_empty`` (assistant text was empty/whitespace). No
                            judge call made; no ``scores``.
``error``                   Cell carried a real error before scoring. ``skip_reason`` is
                            one of ``attack_run_error`` (source ``attack_run.error`` was
                            non-null — payload mirrored to ``error_detail``) or
                            ``target_unparseable`` (target's HTTP wrapper had no
                            recognised refusal signal — wrapper blob in
                            ``error_detail.raw_target_response``). No judge call made.
``judge_refused``           Judge produced a refusal instead of a rubric. No retry
                            (deterministic at temp=0). ``error_detail.raw_judge_output``
                            holds the refusal text.
``parse_error``             Judge output failed to parse after PARSE_RETRY_LIMIT retries.
                            Covers both malformed rubrics AND empty 200 OK replies (no
                            content to parse). ``error_detail`` holds ``raw_judge_output``
                            + ``parse_error`` (+ ``exception`` for empty-reply path).
``transient_error``         429 / 5xx / network / timeout, exhausted transient retries.
                            ``error_detail`` holds the exception payload.
==========================  ==============================================================

Smoke ``.smoke.json`` files sit next to (do not overwrite) the full
``.jsonl`` so iterative smoke-checks don't clobber an in-progress full run.

================================================================================
RESUME / SKIP-IF-EXISTS
================================================================================

Default behaviour skips any cell whose ``source_line_index`` is already present
in the eval JSONL — a partial run resumes cleanly on the next invocation. Pass
``--force`` to wipe the eval file and rescore every selected cell.

================================================================================
DEFAULT JUDGE PROVIDER
================================================================================

Default judge model: DeepSeek V4 Flash

Resolved in this order: ``--judge-provider`` flag → ``JUDGE_PROVIDER`` env →
``judge.DEFAULT_JUDGE_PROVIDER``, and ships ``temperature=0.0`` in its registry row
so the judge stays deterministic.

Provider string → registry row mapping is exact-match against
``ModelConfig.provider`` in ``backend/config/models.py``; resolution lives in
``score_results._resolve_judge_cfg``. Known providers:
``deepseek | moonshot | anthropic | openai | google | xai | together``.

Override examples::

    python score_results.py --judge-provider moonshot   # Kimi K2.5
    JUDGE_PROVIDER=moonshot python score_results.py     # via env

================================================================================
USAGE — run from the ``backend/`` directory
================================================================================

    # 1. FULL RUN — score every cell under results/ that isn't scored yet
    python score_results.py

    # 2. ONE OR MORE MODELS (substring against slug / display_name / provider)
    python score_results.py --model claude                  # claude_sonnet_4_6
    python score_results.py --model anthropic               # same (matches provider)
    python score_results.py --model claude kimi             # space-separated
    python score_results.py --model claude --model kimi     # repeatable
    python score_results.py --model claude,kimi             # comma-separated
    python score_results.py --model claude,kimi --model gpt # mix and match

    # 3. ONE OR MORE METHODS
    python score_results.py --method direct
    python score_results.py --method direct base64          # space-separated
    python score_results.py --method direct --method base64 # repeatable
    python score_results.py --method direct,base64          # comma-separated

    # 4. PROMPT-COUNT VARIATIONS (compose with any of the above)
    #    a) cap total cells across the whole run
    python score_results.py --max-prompts 10
    #    b) cap per category within each (model x method) cell
    python score_results.py --prompts-per-category 1
    #    c) one cell per (model x method) — fastest matrix-coverage smoke
    python score_results.py --smoke

    # 5. COMBINED — typical iteration loop
    python score_results.py --model claude --method direct --max-prompts 1

    # 6. PREVIEW WITHOUT SPENDING TOKENS
    python score_results.py --model claude --method pair --dry-run

    # 7. RESCORE WITH A DIFFERENT JUDGE (default is deepseek)
    python score_results.py --model claude --method direct \\
        --judge-provider moonshot --force

    # 8. CUSTOM I/O DIRECTORIES
    python score_results.py --results-dir ./my_results --eval-dir ./my_evals

================================================================================
FLAGS
================================================================================

==========================  ===============================================
``--model NEEDLE``          substring (model_slug / display_name / provider); repeatable / CSV; OR-match
``--exclude-model NEEDLE``  substring exclusion, repeatable / CSV; default ``["gemini"]``
``--method NAME``           direct | base64 | pair | crescendo; repeatable / CSV
``--prompts-per-category N``  per-(model, method, category) cap
``--max-prompts N``         total cells cap (alias of ``--limit``)
``--limit N``               total cells cap (existing flag, kept)
``--smoke``                 one cell per (model_slug, method)
``--dry-run``               list selected cells and exit, no API calls
``--force``                 truncate + rescore existing eval files
``--judge-provider NAME``   override judge provider
``--results-dir DIR``       source root (default: ``../results``)
``--eval-dir DIR``          eval-output root (default: ``../eval_results``)
==========================  ===============================================

Filter precedence: ``--model`` and ``--method`` narrow the cell list first;
then ``--smoke`` (one per cell) **or** the cap flags (``--max-prompts`` /
``--limit`` / ``--prompts-per-category``) trim further. ``--dry-run``
short-circuits before any API call.

Mutually exclusive (rc=2 with a clear message):
  * ``--smoke`` with ``--max-prompts`` or ``--limit``

================================================================================
EXIT CODES
================================================================================

0 — every selected cell scored or skipped cleanly
1 — at least one cell errored mid-pipeline (judge crashed unexpectedly)
2 — config error: empty rubric, unknown ``--judge-provider``, missing API key,
    nonexistent ``--results-dir``, or mutually-exclusive flags. Also returned
    immediately on the first 401/402/403 from the judge — auth failures halt
    the queue rather than mass-marking remaining cells as failed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("score_results")

import judge  # noqa: E402
from attacks import ATTACK_METHODS  # noqa: E402
from config.models import MODEL_REGISTRY, ModelConfig  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_EVAL_DIR = Path(__file__).parent.parent / "eval_results"

# Max source files (full mode) / smoke cells (smoke mode) processed in
# parallel. Each worker owns its own output file, so file-write contention is
# nil — the only reason to cap is provider-side rate limits. 8 is a safe
# default for DeepSeek/Kimi; lower for stricter providers via --concurrency.
DEFAULT_JUDGE_CONCURRENCY = 8

# Default model exclusions applied to ``--results-dir`` discovery. Substring
# match (case-insensitive) against ``model_slug`` / ``display_name`` /
# ``provider`` — same matcher as ``--model``. Gemini is excluded by default:
# its safety filter rejects the bio rubric prompt itself, producing wrapped
# 4xx blobs that even the synthetic-refusal short-circuit can't grade
# meaningfully. Override with ``--exclude-model ''`` to include everything.
# Applied at file-discovery time so EVERY mode (full, --smoke, --dry-run)
# inherits the exclusion.
DEFAULT_MODEL_EXCLUSIONS: tuple[str, ...] = ("gemini",)


# ---------------------------------------------------------------------------
# Slug + filter helpers
# ---------------------------------------------------------------------------


def _model_slug(display_name: str) -> str:
    """Mirror matrix_runner's slug convention.

    ``"Claude Sonnet 4.6"`` → ``"claude_sonnet_4_6"``. Lowercase, then collapse
    space, dot, and hyphen to underscore. Used to cross-reference the registry
    against the path segment matrix_runner writes.
    """
    s = display_name.lower()
    for ch in (" ", ".", "-"):
        s = s.replace(ch, "_")
    return s


# Module-level slug→cfg map. Built once at import time so ``_model_matches``
# avoids the O(N) ``next(...)`` scan on every call. MODEL_REGISTRY is a
# constant list so the dict never goes stale.
_SLUG_TO_CFG: dict[str, ModelConfig] = {
    _model_slug(c.display_name): c for c in MODEL_REGISTRY
}


def _model_matches(slug: str, needle: str | None) -> bool:
    """Substring match against slug / display_name / provider.

    Mirrors ``matrix_runner.py --model NEEDLE`` selection: a single needle
    matches a row if any of its three identities contain the needle
    (case-insensitive). ``None`` / empty needle matches everything.
    """
    if not needle:
        return True
    n = needle.lower()
    if n in slug.lower():
        return True
    cfg = _SLUG_TO_CFG.get(slug)
    if cfg is None:
        return False
    return n in cfg.display_name.lower() or n in cfg.provider.lower()


# ---------------------------------------------------------------------------
# Cell loading
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cell:
    """One attack-run record located within a source JSONL file.

    ``line_idx`` is the 0-based offset of the record inside ``src`` and is the
    resume key — we never re-score a cell whose ``source_line_index`` is
    already on disk in the corresponding eval JSONL.
    """

    src: Path
    line_idx: int
    model_slug: str
    method: str
    record: dict[str, Any]


def _flatten_csv(values: Any) -> list[str]:
    """Flatten any of the argparse list shapes we use, splitting on commas.

    Handles both:
      * flat list (``action='extend'``)        — e.g. ``['a', 'b,c', 'd']``
      * nested list (``action='append' + nargs='+'``) — e.g. ``[['a','b,c'],['d']]``

    Empty strings are dropped, so ``--exclude-model ''`` flattens to ``[]``.
    """
    if not values:
        return []
    out: list[str] = []

    def _add(v: Any) -> None:
        if isinstance(v, list):
            for item in v:
                _add(item)
        elif isinstance(v, str):
            for part in v.split(","):
                p = part.strip()
                if p:
                    out.append(p)

    for v in values:
        _add(v)
    return out


def _filter_files(
    results_dir: Path,
    *,
    model_needles: list[str] | tuple[str, ...] = (),
    methods: list[str] | tuple[str, ...] = (),
    exclude_needles: tuple[str, ...] | list[str] = (),
) -> list[tuple[str, str, Path]]:
    """Walk matrix_runner JSONL files under ``results_dir`` and apply filters.

    Returns ``[(model_slug, method, path), ...]`` sorted by path.

    Strict matrix_runner contract:
      * extension must be ``.jsonl`` (validate_attacks ``.json`` files ignored)
      * path must match ``{model_slug}/{method}/<run>.jsonl`` (3 segments
        relative to ``results_dir``); shorter paths are skipped silently

    Filter semantics:
      * ``methods`` empty       → all methods accepted
      * ``methods`` non-empty   → method segment must equal one entry exactly
      * ``model_needles`` empty → all slugs accepted
      * ``model_needles`` set   → at least ONE needle must substring-match
                                  via ``_model_matches`` (OR semantics)
      * ``exclude_needles``     → if ANY exclusion substring-matches the
                                  slug, the cell is dropped (overrides
                                  include matches — exclusion always wins)
    """
    out: list[tuple[str, str, Path]] = []
    for p in sorted(results_dir.rglob("*.jsonl")):
        rel = p.relative_to(results_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        model_slug, method_seg = parts[0], parts[1]
        if methods and method_seg not in methods:
            continue
        if model_needles and not any(
            _model_matches(model_slug, n) for n in model_needles
        ):
            continue
        if any(_model_matches(model_slug, n) for n in exclude_needles):
            continue
        out.append((model_slug, method_seg, p))
    # Heads-up if validate_attacks-style files exist under the same root —
    # they're being skipped, not silently scored. Useful when a user expected
    # them to count.
    stray_json = [
        p for p in results_dir.rglob("*.json")
        if not p.name.endswith(".smoke.json")
    ]
    if stray_json:
        logger.info(
            "Ignoring %d non-JSONL file(s) under %s "
            "(validate_attacks output is not a scoring input).",
            len(stray_json),
            results_dir,
        )
    return out


def _iter_cells(files: list[tuple[str, str, Path]]) -> Iterable[Cell]:
    """Yield every cell from the selected source files in path + line order.

    Malformed lines are logged and skipped — a single bad line never aborts
    the run, mirroring matrix_runner's "matrix never aborts mid-run" guarantee.
    """
    for model_slug, method, src in files:
        with src.open() as fh:
            for line_idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSONL line %d in %s: %s",
                        line_idx,
                        src,
                        exc,
                    )
                    continue
                yield Cell(
                    src=src,
                    line_idx=line_idx,
                    model_slug=model_slug,
                    method=method,
                    record=record,
                )


# ---------------------------------------------------------------------------
# Cell-level filters / samplers
# ---------------------------------------------------------------------------


def _apply_per_category_cap(
    cells: list[Cell], n: int | None
) -> list[Cell]:
    """Keep first ``n`` cells per (model_slug, method, category).

    Mirrors ``matrix_runner.py --prompts-per-category``. Order within each
    bucket follows source-file iteration order, which is sorted-path then
    line index — deterministic across runs.
    """
    if n is None:
        return cells
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    out: list[Cell] = []
    for c in cells:
        category = str(c.record.get("category", "<unknown>"))
        key = (c.model_slug, c.method, category)
        if counts[key] >= n:
            continue
        counts[key] += 1
        out.append(c)
    return out


def _smoke_sample(cells: list[Cell]) -> list[Cell]:
    """One cell per (model_slug, method) — fastest way to exercise the matrix."""
    seen: dict[tuple[str, str], Cell] = {}
    for c in cells:
        key = (c.model_slug, c.method)
        if key not in seen:
            seen[key] = c
    return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------------------
# Resume / skip-if-exists
# ---------------------------------------------------------------------------


def _existing_indices(out_path: Path) -> set[int]:
    """Source-line indices already present in the eval JSONL.

    Lets a re-run resume after a crash without rescoring completed cells.
    Malformed eval lines are ignored — they'll be re-scored, which is safe.
    """
    if not out_path.exists():
        return set()
    indices: set[int] = set()
    with out_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                idx = json.loads(line).get("source_line_index")
            except json.JSONDecodeError:
                continue
            if isinstance(idx, int):
                indices.add(idx)
    return indices


# ---------------------------------------------------------------------------
# Judge config
# ---------------------------------------------------------------------------


def _resolve_judge_cfg(provider: str) -> ModelConfig:
    cfg = next((c for c in MODEL_REGISTRY if c.provider == provider), None)
    if cfg is None:
        known = sorted(c.provider for c in MODEL_REGISTRY)
        raise EnvironmentError(
            f"Unknown judge provider {provider!r}. Known: {known}."
        )
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score matrix-runner JSONL with the bio-aware LLM judge. "
            "Selection flags mirror matrix_runner.py."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe + rescore eval JSONL files even if they exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total cells scored across the whole run (alias of --max-prompts).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        dest="max_prompts",
        help="Cap total cells scored across the whole run (alias of --limit).",
    )
    parser.add_argument(
        "--prompts-per-category",
        type=int,
        default=None,
        dest="prompts_per_category",
        help="Keep first N cells per (model_slug, method, category).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Source root: matrix_runner JSONL tree.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Output root: eval JSONL tree (mirrors source layout).",
    )
    # ``type=str.lower`` on string args that name slugs / methods / providers
    # so the user can type any case (e.g. ``--judge-provider MOONSHOT``,
    # ``--method DIRECT``) and it still matches. Path args (``--results-dir``
    # / ``--eval-dir``) stay untouched — POSIX paths are case-sensitive and
    # silently lowercasing them would break valid mixed-case directories.
    parser.add_argument(
        "--judge-provider",
        type=str.lower,
        default=None,
        help=(
            "Override the judge LLM provider. Resolution: flag → "
            "JUDGE_PROVIDER env → judge.DEFAULT_JUDGE_PROVIDER (deepseek). "
            "Must match a ModelConfig.provider in backend/config/models.py. "
            "Case-insensitive (input is lowercased before lookup)."
        ),
    )
    # ``action='extend'`` + ``nargs='+'`` lets each occurrence of the flag
    # accept one or more space-separated values, and multiple occurrences
    # extend a single flat list. Combined with ``_flatten_csv`` this covers:
    #   --model a               --model a b               (space-separated)
    #   --model a --model b     --model a,b               (CSV)
    #   --model a,b c --model d e,f                       (mix)
    # ``default=[]`` is required because ``extend`` calls ``.extend`` on the
    # existing attribute; ``None`` would crash. The list is fresh per
    # ``_parse_args`` call (parser created inside the function), so the
    # mutable-default footgun does not apply here.
    parser.add_argument(
        "--model",
        action="extend",
        nargs="+",
        type=str.lower,
        default=[],
        metavar="NEEDLE",
        help=(
            "Substring match (case-insensitive) against model_slug / "
            "display_name / provider. Accepts space-separated, comma-"
            "separated, and repeated forms: "
            "``--model claude kimi`` / ``--model claude,kimi`` / "
            "``--model claude --model kimi``. OR-match across needles."
        ),
    )
    parser.add_argument(
        "--method",
        action="extend",
        nargs="+",
        type=str.lower,
        default=[],
        metavar="NAME",
        help=(
            "Attack method filter. Choices: "
            f"{sorted(ATTACK_METHODS.keys())}. Accepts space-separated, "
            "comma-separated, and repeated forms: ``--method direct base64`` / "
            "``--method direct,base64`` / ``--method direct --method base64``. "
            "Validated at parse time."
        ),
    )
    # --exclude-model keeps ``action='append' + nargs='+'`` so we can still
    # tell "user did not pass the flag" (None → defaults apply) from "user
    # passed an empty value to disable defaults" ([['']] → flatten to []).
    # Switching to ``extend`` would collapse both into ``[]`` and lose that
    # signal. The recursive ``_flatten_csv`` handles the nested list shape.
    parser.add_argument(
        "--exclude-model",
        action="append",
        nargs="+",
        type=str.lower,
        default=None,
        dest="exclude_model",
        metavar="NEEDLE",
        help=(
            "Substring exclusion against model_slug / display_name / provider. "
            "Accepts space-separated, comma-separated, and repeated forms: "
            "``--exclude-model gemini grok`` / ``--exclude-model gemini,grok`` "
            "/ ``--exclude-model gemini --exclude-model grok``. Defaults to "
            f"{list(DEFAULT_MODEL_EXCLUSIONS)!r}; pass --exclude-model '' to "
            "score every model. Explicit values REPLACE the defaults."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="List selected cells (one line per cell) and exit. No API calls.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Sample one cell per (model_slug, method).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_JUDGE_CONCURRENCY,
        help=(
            "Max parallel judge workers. One worker owns one output file "
            f"end-to-end (no write contention). Default {DEFAULT_JUDGE_CONCURRENCY}. "
            "Lower this if the provider rate-limits aggressively."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-cell scorer
# ---------------------------------------------------------------------------


def _is_crescendo_record(record: dict[str, Any]) -> bool:
    """Detect matrix_runner's crescendo schema.

    Crescendo records carry split ``attacker_conversation`` /
    ``defender_conversation`` lists and an ``objective_prompt`` string —
    nothing else uses that shape, so presence of both is a reliable signal.
    """
    return (
        isinstance(record.get("defender_conversation"), list)
        and "objective_prompt" in record
    )


def _normalize_crescendo_record(record: dict[str, Any]) -> dict[str, Any]:
    """Crescendo record → standard attack_run shape.

    Reads exactly two fields:
      * top-level ``objective_prompt``
      * ``defender_conversation[-1]["text"]``
    """
    defender_turns = record.get("defender_conversation") or []
    convo: list[dict[str, Any]] = []
    if defender_turns and isinstance(defender_turns[-1], dict):
        text = defender_turns[-1].get("text", "")
        convo = [
            {
                "role": "assistant",
                "converted_value": text,
                "original_value": text,
            }
        ]
    return {
        "objective": record.get("objective_prompt", ""),
        "conversation": convo,
        "error": None,
    }


def _record_prompt_id(record: dict[str, Any]) -> Any:
    """Top-level ``prompt_id`` for standard records; ``metadata.prompt_id`` for crescendo."""
    pid = record.get("prompt_id")
    if pid is not None:
        return pid
    md = record.get("metadata")
    if isinstance(md, dict):
        return md.get("prompt_id")
    return None


def _record_category(record: dict[str, Any]) -> str:
    """Top-level ``category`` for standard records; ``metadata.category`` for crescendo."""
    cat = record.get("category")
    if cat:
        return str(cat)
    md = record.get("metadata")
    if isinstance(md, dict):
        return str(md.get("category") or "?")
    return "?"


async def _score_cell(
    cell: Cell,
    judge_target: Any,
    judge_cfg: ModelConfig,
) -> dict[str, Any]:
    """Run the judge against a single cell and assemble the eval-line dict.

    For crescendo cells the source record is normalised to the standard
    attack_run shape (``_normalize_crescendo_record``) before scoring. The
    eval line still echoes the **original** record under ``attack_run`` so
    nothing is lost downstream — only the judge sees the trimmed view.
    """
    record_for_judge = (
        _normalize_crescendo_record(cell.record)
        if _is_crescendo_record(cell.record)
        else cell.record
    )
    eval_run = await judge.score_attack_run(
        record_for_judge, judge_target, judge_cfg
    )
    return {
        "source_line_index": cell.line_idx,
        "prompt_id": _record_prompt_id(cell.record),
        "attack_run": cell.record,
        "eval_run": eval_run,
    }


# ---------------------------------------------------------------------------
# Concurrent workers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _WorkerStats:
    """Per-worker counters aggregated by main() after gather."""

    n_scored: int = 0
    n_skipped: int = 0
    n_errored: int = 0
    status_counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def bump_status(self, status: str | None) -> None:
        key = status or "<none>"
        self.status_counts[key] = self.status_counts.get(key, 0) + 1


def _log_eval_outcome(prefix: str, display: Any, eval_run: dict[str, Any]) -> None:
    status = eval_run.get("status")
    asr = (eval_run.get("scores") or {}).get("ASR")
    attempts = eval_run.get("attempts")
    if status == judge.STATUS_OK and asr is not None:
        logger.info("%s: %s [ASR=%.3f attempts=%s]", prefix, display, asr, attempts)
    elif status == judge.STATUS_FILTER:
        signal = (eval_run.get("judge") or {}).get("synthetic_signal", "?")
        logger.info(
            "%s (filter: %s): %s [ASR=%.3f]",
            prefix, signal, display, asr if asr is not None else 0.0,
        )
    elif status == judge.STATUS_EMPTY_RESPONSE:
        logger.info(
            "%s (empty_response: %s): %s",
            prefix, eval_run.get("skip_reason"), display,
        )
    elif status == judge.STATUS_ERROR:
        logger.warning(
            "%s (error: %s): %s",
            prefix, eval_run.get("skip_reason"), display,
        )
    else:
        logger.warning(
            "%s (%s, attempts=%s): %s", prefix, status, attempts, display
        )


async def _full_worker(
    *,
    src: Path,
    src_cells: list[Cell],
    results_dir: Path,
    eval_dir: Path,
    judge_target: Any,
    judge_cfg: ModelConfig,
    force: bool,
    sem: asyncio.Semaphore,
    halt: asyncio.Event,
    auth_exc_holder: list[BaseException],
) -> _WorkerStats:
    """One worker per source JSONL — owns its output file end-to-end.

    Sequentially scores every cell in this source and appends to the
    matching ``eval_dir`` JSONL. Honours ``force`` (truncate first), the
    ``existing`` resume set (skip lines already scored), and ``halt`` (drop
    out the moment any other worker hits an auth failure).
    """
    stats = _WorkerStats()
    if halt.is_set():
        return stats
    rel = src.relative_to(results_dir)
    out_path = eval_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if force and out_path.exists():
        out_path.unlink()
        logger.info("--force: wiped existing %s", rel)
    existing = _existing_indices(out_path)

    async with sem:
        if halt.is_set():
            return stats
        with out_path.open("a") as out_fh:
            for c in src_cells:
                if halt.is_set():
                    break
                if c.line_idx in existing:
                    stats.n_skipped += 1
                    continue
                try:
                    line_obj = await _score_cell(c, judge_target, judge_cfg)
                except judge.JudgeAuthError as exc:
                    halt.set()
                    auth_exc_holder.append(exc)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "scoring failed for %s:%d — %r", rel, c.line_idx, exc
                    )
                    stats.n_errored += 1
                    continue
                out_fh.write(json.dumps(line_obj, default=str) + "\n")
                out_fh.flush()
                eval_run = line_obj["eval_run"]
                stats.bump_status(eval_run.get("status"))
                _log_eval_outcome(
                    "scored", f"{rel}:{c.line_idx}", eval_run
                )
                stats.n_scored += 1
    return stats


async def _smoke_worker(
    *,
    cell: Cell,
    results_dir: Path,
    eval_dir: Path,
    judge_target: Any,
    judge_cfg: ModelConfig,
    force: bool,
    sem: asyncio.Semaphore,
    halt: asyncio.Event,
    auth_exc_holder: list[BaseException],
) -> _WorkerStats:
    """One worker per smoke cell — writes its own ``.smoke.json``."""
    stats = _WorkerStats()
    if halt.is_set():
        return stats
    rel = cell.src.relative_to(results_dir)
    out_path = (eval_dir / rel).with_suffix(".smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display = out_path.relative_to(eval_dir)

    if out_path.exists() and not force:
        logger.info("skip (exists): %s", display)
        stats.n_skipped += 1
        return stats

    async with sem:
        if halt.is_set():
            return stats
        try:
            line_obj = await _score_cell(cell, judge_target, judge_cfg)
        except judge.JudgeAuthError as exc:
            halt.set()
            auth_exc_holder.append(exc)
            return stats
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "scoring failed for %s:%d — %r", rel, cell.line_idx, exc
            )
            stats.n_errored += 1
            return stats

        out_path.write_text(json.dumps(line_obj, indent=2, default=str))
        eval_run = line_obj["eval_run"]
        stats.bump_status(eval_run.get("status"))
        _log_eval_outcome("smoke", display, eval_run)
        stats.n_scored += 1
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # --- Mutually exclusive flag validation ---------------------------------
    cap = args.max_prompts if args.max_prompts is not None else args.limit
    if args.smoke and cap is not None:
        logger.error(
            "--smoke and --max-prompts/--limit are mutually exclusive."
        )
        return 2

    # --- Rubric + judge config ---------------------------------------------
    try:
        version = judge.prompt_version()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Rubric version: %s", version)

    # Same case-insensitive contract as the flag: env var input is
    # lowercased so JUDGE_PROVIDER=Moonshot still resolves.
    provider = (
        args.judge_provider
        or os.getenv("JUDGE_PROVIDER")
        or judge.DEFAULT_JUDGE_PROVIDER
    ).strip().lower()
    try:
        judge_cfg = _resolve_judge_cfg(provider)
    except EnvironmentError as exc:
        logger.error("%s", exc)
        return 2
    judge_cfg = dataclasses.replace(
        judge_cfg, max_completion_tokens=judge.JUDGE_MAX_COMPLETION_TOKENS
    )

    results_dir: Path = args.results_dir
    eval_dir: Path = args.eval_dir
    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        return 2

    # --- Discover + filter --------------------------------------------------
    # Resolve --exclude-model: omitted → DEFAULT_MODEL_EXCLUSIONS; explicit
    # value(s) REPLACE the defaults; empty string disables exclusions entirely.
    # The exclusion is applied here (file discovery), so it covers every
    # downstream mode uniformly: full, --smoke, --dry-run.
    if args.exclude_model is None:
        exclude_needles: list[str] = list(DEFAULT_MODEL_EXCLUSIONS)
    else:
        # Allow comma-split exclusions too: ``--exclude-model gemini,grok``.
        # ``[]`` (e.g. from a single ``--exclude-model ''``) means "disable
        # default exclusions" — we keep that behaviour.
        exclude_needles = _flatten_csv(args.exclude_model)
        if args.exclude_model and not exclude_needles:
            # Explicit empty arg(s) only — defaults are intentionally cleared.
            exclude_needles = []
    if exclude_needles:
        logger.info("Excluding models matching: %s", exclude_needles)

    # --model / --method now accept multiple values via repetition or CSV.
    model_needles = _flatten_csv(args.model)
    methods = _flatten_csv(args.method)

    # Validate methods at parse time — argparse choices= can't run when the
    # arg accepts comma-split lists, so we check ourselves.
    unknown_methods = sorted(set(methods) - set(ATTACK_METHODS))
    if unknown_methods:
        logger.error(
            "Unknown --method value(s): %s. Known: %s.",
            unknown_methods,
            sorted(ATTACK_METHODS),
        )
        return 2

    files = _filter_files(
        results_dir,
        model_needles=model_needles,
        methods=methods,
        exclude_needles=exclude_needles,
    )
    if not files:
        logger.warning(
            "No JSONL files under %s match filters "
            "(model=%s method=%s exclude=%s).",
            results_dir,
            model_needles or "<all>",
            methods or "<all>",
            exclude_needles,
        )
        return 0

    cells: list[Cell] = list(_iter_cells(files))
    cells = _apply_per_category_cap(cells, args.prompts_per_category)

    if args.smoke:
        cells = _smoke_sample(cells)
        logger.info(
            "Smoke sample: %d (model_slug x method) cell(s).", len(cells)
        )
    elif cap is not None:
        cells = cells[:cap]

    if not cells:
        logger.warning("No cells selected after filters.")
        return 0

    # --- Dry run: list and exit, no API calls -------------------------------
    if args.dry_run:
        logger.info(
            "Dry run — would score %d cell(s); no API calls.", len(cells)
        )
        for c in cells:
            # _record_prompt_id / _record_category hoist crescendo fields
            # from metadata so the dry-run line matches what would actually
            # get written to the eval JSONL.
            print(
                f"{c.src}:{c.line_idx}\t{c.model_slug}/{c.method}\t"
                f"{_record_category(c.record)}\t"
                f"{_record_prompt_id(c.record) or '?'}"
            )
        return 0

    # No PyRIT init, no build_target — the judge is called directly via
    # litellm. We pass judge_cfg through as the opaque "target" for
    # judge.score_attack_run / judge.call_judge.
    judge_target = judge_cfg
    logger.info(
        "Judge LLM: %s (%s) max_completion_tokens=%d",
        judge_cfg.display_name,
        judge_cfg.provider,
        judge_cfg.max_completion_tokens,
    )

    # --- Score loop (concurrent — one worker per output file) --------------
    # Each worker owns one output file end-to-end:
    #   * full mode: one .jsonl per (model_slug, method) source — worker
    #     processes its source's cells sequentially and appends to the file.
    #   * smoke mode: one .smoke.json per cell — each cell IS its own worker.
    # Workers across different output files run in parallel under a single
    # ``asyncio.Semaphore`` cap. No two workers ever touch the same file, so
    # we never need a write lock. JudgeAuthError on any worker sets a shared
    # ``halt`` event that other workers honour at their next loop boundary.
    sem = asyncio.Semaphore(args.concurrency)
    halt = asyncio.Event()
    auth_exc_holder: list[BaseException] = []

    if args.smoke:
        workers = [
            _smoke_worker(
                cell=c,
                results_dir=results_dir,
                eval_dir=eval_dir,
                judge_target=judge_target,
                judge_cfg=judge_cfg,
                force=args.force,
                sem=sem,
                halt=halt,
                auth_exc_holder=auth_exc_holder,
            )
            for c in cells
        ]
    else:
        by_src: dict[Path, list[Cell]] = defaultdict(list)
        for c in cells:
            by_src[c.src].append(c)
        workers = [
            _full_worker(
                src=src,
                src_cells=src_cells,
                results_dir=results_dir,
                eval_dir=eval_dir,
                judge_target=judge_target,
                judge_cfg=judge_cfg,
                force=args.force,
                sem=sem,
                halt=halt,
                auth_exc_holder=auth_exc_holder,
            )
            for src, src_cells in by_src.items()
        ]

    logger.info(
        "Dispatching %d worker(s) at concurrency=%d.",
        len(workers),
        args.concurrency,
    )
    results: list[_WorkerStats] = await asyncio.gather(*workers)

    if auth_exc_holder:
        logger.error(
            "Judge auth/quota error — halting pipeline. %s",
            auth_exc_holder[0],
        )
        return 2

    n_scored = sum(r.n_scored for r in results)
    n_skipped_exists = sum(r.n_skipped for r in results)
    n_errored = sum(r.n_errored for r in results)
    status_counts: dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r.status_counts.items():
            status_counts[k] += v

    logger.info(
        "summary: scored=%d skipped_existing=%d errored=%d total_selected=%d "
        "status_breakdown=%s",
        n_scored,
        n_skipped_exists,
        n_errored,
        len(cells),
        dict(status_counts),
    )
    return 0 if n_errored == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
