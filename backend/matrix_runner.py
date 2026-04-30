"""Matrix Experiment Runner — iterate Model x Method x Category x Prompt.

Orchestrates the full test matrix across all available models and attack
methods. Each cell is appended as one JSON line to:

    <output-root>/<model_slug>/<method>/<N>_<ddmmyy>_<dataset>.jsonl

where N is the total prompt count for the run, ddmmyy is today's date, and
dataset is the prompt-file stem (e.g. dummy_prompts). Default output-root is
``../results`` so files land at the project root, not under backend/.

Models without an API key in .env are skipped automatically. Individual cell
failures are recorded in the JSONL rather than aborting the matrix — re-running
the same command resumes by skipping cells already on disk under the root.

================================================================================
USAGE — run all commands from the ``backend/`` directory
================================================================================

# Full matrix (every available model, every attack, every prompt)
python matrix_runner.py

# One model (substring-matches provider OR display name)
python matrix_runner.py --model claude
python matrix_runner.py --model kimi

# One attack method
python matrix_runner.py --method direct      # plain prompt
python matrix_runner.py --method base64      # base64-encoded prompt
python matrix_runner.py --method pair        # multi-turn, adversary required
python matrix_runner.py --method crescendo   # multi-turn, adversary required

# Prompt-count knobs (compose with any flag above)
python matrix_runner.py --max-prompts 1
python matrix_runner.py --prompts-per-category 1
python matrix_runner.py --prompt-file prompts/dummy_prompts.json

# One model + one method + one prompt (cheapest smoke)
python matrix_runner.py --model deepseek --method crescendo --max-prompts 1 \
    --prompt-file ../prompts/mock_prompts.json

# Adversary override for pair/crescendo (default: moonshot/Kimi). pair and
# crescendo refuse to run when target == adversary, so override when the
# selected --model matches the default adversary.
ADVERSARY_PROVIDER=deepseek python matrix_runner.py --model kimi --method crescendo

# Custom output root
python matrix_runner.py --output-root ./my_results

================================================================================
CRESCENDO DEBUG MODE — separate path, self-play with full transcript logging
================================================================================
Writes to ../results/<defender_slug>/crescendo/ — same tree shape as the matrix
runner, so traces sit alongside the standard direct/base64/pair/crescendo cells.
Resume by pointing --crescendo-resume-from-jsonl at the same .jsonl path.

# Full bench, Kimi attacks Claude
python matrix_runner.py --crescendo-debug --crescendo-debug-full \
    --crescendo-kimi-attacks-anthropic --prompt-file ../prompts/prompts_long.json

# One-prompt smoke (Kimi attacker -> DeepSeek defender, default pairing)
python matrix_runner.py --crescendo-debug --prompt-file ../prompts/mock_prompts.json

Pairings (pick at most one; default is kimi-attacker -> deepseek-defender):
    --crescendo-kimi-attacks-anthropic   Kimi -> Claude
    --crescendo-openai-both              Kimi -> OpenAI
    --crescendo-anthropic-both           Claude -> Kimi
    --crescendo-moonshot-both            Kimi -> Kimi

Other crescendo-debug flags: --crescendo-debug-full (load every prompt),
--crescendo-bench {long|short}, --debug-output PATH, --attacker-provider,
--defender-provider, --crescendo-resume-from-jsonl PATH. See CLAUDE.md ->
"Crescendo debug: resume progress" for the append-to-same-file pattern.

python matrix_runner.py --crescendo-debug --crescendo-debug-full --defender-provider anthropic_opus --prompt-file ../prompts/prompts_short.json   

================================================================================
FLAGS (standard matrix)
================================================================================
--prompt-file PATH         JSON prompt file (default: prompts/prompts_long.json)
--max-prompts N            keep first N prompts across all categories
--prompts-per-category N   keep first N prompts within each category
--method NAME              direct | base64 | pair | crescendo (default: all)
--model NEEDLE             filter buildable models by provider/display substring
--output-root DIR          results tree root (default: ../results)
--mode LABEL               written to each record's "mode" field (default: smoke)

================================================================================
RESUME
================================================================================
Cells flush one line at a time. Re-running the same command skips cells already
on disk under --output-root. To force a redo, delete the matching JSONL file.
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import logging
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

load_dotenv()

from pyrit.memory import CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

# Import order matters: try absolute (`backend.X`) first when CWD==repo root.
# Bare imports (`from attacks ...`) work when CWD==backend/, but from repo root
# they would resolve to stale root-level attacks.py / config/ shadow copies.
try:  # CWD == repo root (e.g. `python backend/matrix_runner.py ...`)
    from backend.attacks import (
        ATTACK_METHODS,
        METHODS_REQUIRING_ADVERSARY,
        PLACEHOLDER_SCORER,
        PLACEHOLDER_SCORER_METHODS,
    )
    from backend.config.models import ModelConfig, build_target, get_available_models
    from backend.crescendo_debug import (
        DEFAULT_DEBUG_ATTACKER_PROVIDER,
        DEFAULT_DEBUG_DEFENDER_PROVIDER,
        DEFAULT_DEBUG_OUTPUT_DIR,
        DEFAULT_DEBUG_PROMPT_FILES,
        DEFAULT_FULL_BENCH_PROMPT_FILE,
        DEFAULT_SHORT_BENCH_PROMPT_FILE,
        load_completed_prompt_ids_from_jsonl,
        run_crescendo_anthropic_as_both_roles,
        run_crescendo_debug_smoke_test,
        run_crescendo_kimi_attacks_anthropic,
        run_crescendo_moonshot_as_both_roles,
        run_crescendo_openai_as_both_roles,
    )
except ModuleNotFoundError:  # CWD == backend/  (e.g. `cd backend && python matrix_runner.py ...`)
    from attacks import (
        ATTACK_METHODS,
        METHODS_REQUIRING_ADVERSARY,
        PLACEHOLDER_SCORER,
        PLACEHOLDER_SCORER_METHODS,
    )
    from config.models import ModelConfig, build_target, get_available_models
    from crescendo_debug import (
        DEFAULT_DEBUG_ATTACKER_PROVIDER,
        DEFAULT_DEBUG_DEFENDER_PROVIDER,
        DEFAULT_DEBUG_OUTPUT_DIR,
        DEFAULT_DEBUG_PROMPT_FILES,
        DEFAULT_FULL_BENCH_PROMPT_FILE,
        DEFAULT_SHORT_BENCH_PROMPT_FILE,
        load_completed_prompt_ids_from_jsonl,
        run_crescendo_anthropic_as_both_roles,
        run_crescendo_debug_smoke_test,
        run_crescendo_kimi_attacks_anthropic,
        run_crescendo_moonshot_as_both_roles,
        run_crescendo_openai_as_both_roles,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Retry parameters for rate-limit / transient errors
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 5

# Loop in run_single_cell uses range(1, MAX_RETRIES + 1); a value of 0 would
# yield an empty range and silently skip every cell, breaking the "matrix never
# aborts mid-run" guarantee. Hard-fail at import time if a future edit lowers it.
assert MAX_RETRIES >= 1, "MAX_RETRIES must be ≥ 1 to guarantee at least one attempt per cell."

# Default cap for victim completion tokens (HarmBench: 512). Per-model
# overrides live in ``ModelConfig.victim_max_tokens`` (e.g. Kimi 2048, Gemini 2048).
# HarmBench reports that varying this can shift ASR by up to 30%, so do not
# lower without recording the change in the run metadata. The adversary
# target (PAIR/Crescendo) is intentionally built without this cap — it needs
# headroom to draft jailbreaks.
MAX_TARGET_COMPLETION_TOKENS = 512

# Substrings that indicate a provider blocked the request via content-filter
# or safety machinery (rather than a generic transient error). Matched on the
# stringified exception, case-insensitively. Keep this list conservative —
# false positives would mask real bugs.
SAFETY_MARKERS: tuple[str, ...] = (
    "content_filter",
    "content filter",
    "content_policy",
    "content policy",
    "responsible_ai_policy",
    "blocked_by_safety",
    "safety_block",
    "safety filter",
    "prompt was blocked",
    "violates our usage policies",
    "violates the usage policies",
)


def _classify_safety(exc: BaseException | None) -> tuple[bool, str | None]:
    """Tag an exception as a content-filter / safety block when recognizable."""
    if exc is None:
        return False, None
    msg = str(exc)
    lower = msg.lower()
    for marker in SAFETY_MARKERS:
        if marker in lower:
            return True, msg
    return False, None


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(
    path: str,
) -> dict[str, list[dict | str]]:
    """Load categorized prompts from a JSON file.

    Supports two schemas:
    - **Rich (v1):** ``{schema_version: 1, categories: {cat: [{prompt_id, prompt_text, ...}]}}``
    - **Flat (legacy):** ``{cat: [str, ...]}``  (backward compat for mock prompts / tests)

    Args:
        path: Path to the prompts JSON file.

    Returns:
        Dict mapping category names to lists of prompt objects (dicts) or
        plain strings (legacy).
    """
    with open(path) as fh:
        doc = json.load(fh)

    if "schema_version" in doc:
        version = doc["schema_version"]
        if version != 1:
            raise ValueError(f"Unsupported prompt schema_version: {version}")
        categories = doc.get("categories")
        if not isinstance(categories, dict):
            raise ValueError("Rich prompt schema must include 'categories' as an object")
        logger.info("Loaded rich prompt schema v1 from %s", path)
        return categories

    logger.info("Loaded legacy flat prompt schema from %s", path)
    return doc


def limit_prompts(
    prompts_by_category: dict[str, list[dict | str]],
    max_prompts: int | None,
) -> dict[str, list[dict | str]]:
    """Keep the first N prompts across categories, preserving category order."""
    if max_prompts is None:
        return prompts_by_category
    if max_prompts < 1:
        raise ValueError("--max-prompts must be at least 1")

    remaining = max_prompts
    limited: dict[str, list[dict | str]] = {}
    for category, prompts in prompts_by_category.items():
        if remaining <= 0:
            break
        selected = prompts[:remaining]
        if selected:
            limited[category] = selected
            remaining -= len(selected)
    return limited


def limit_per_category(
    prompts_by_category: dict[str, list[dict | str]],
    per_category: int | None,
) -> dict[str, list[dict | str]]:
    """Keep the first N prompts in each category. ``--prompts-per-category 1``
    is the natural smoke-test setting: one prompt per category per cell.
    """
    if per_category is None:
        return prompts_by_category
    if per_category < 1:
        raise ValueError("--prompts-per-category must be at least 1")
    return {
        cat: prompts[:per_category]
        for cat, prompts in prompts_by_category.items()
        if prompts[:per_category]
    }


# Attack factories live in attacks.py and are imported as ATTACK_METHODS above.


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    model: str
    method: str
    category: str
    prompt: str
    status: str
    conversation_id: str | None
    elapsed_seconds: float
    prompt_id: str | None = None
    adversary: str | None = None
    scorer: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    # True when the API blocked the request via content-filter / safety
    # machinery (recognized by SAFETY_MARKERS). safety_message carries the
    # provider's raw error string so the report can quote it verbatim.
    content_filter: bool = False
    safety_message: str | None = None
    # Inlined transcript so the JSONL is self-sufficient — the IN_MEMORY
    # PyRIT store dies on process exit, so without this the actual model
    # output would be unrecoverable. final_response is the last assistant
    # turn (most-used field); conversation is every piece in order.
    final_response: str | None = None
    conversation: list[dict] | None = None


def _extract_transcript(
    conversation_id: str,
) -> tuple[list[dict] | None, str | None]:
    """Pull a conversation from CentralMemory and serialize it for JSONL.

    Returns ``(conversation_list, final_assistant_text)``. Both may be None
    if the conversation isn't retrievable (e.g. PyRIT internals changed)
    — we never raise here because the cell already succeeded; losing the
    transcript should not turn an OK result into a failure.
    """
    try:
        memory = CentralMemory.get_memory_instance()
        pieces = memory.get_message_pieces(conversation_id=conversation_id)
    except Exception as exc:  # noqa: BLE001 — defensive; transcript is best-effort
        logger.warning("Could not load transcript for %s: %r", conversation_id, exc)
        return None, None

    conversation: list[dict] = []
    for p in pieces:
        # PyRIT MessagePiece stores role privately (_role) and exposes it
        # via the api_role property — there's no plain `role` attribute.
        conversation.append({
            "role": getattr(p, "api_role", None),
            "original_value": getattr(p, "original_value", None),
            "converted_value": getattr(p, "converted_value", None),
            "data_type": getattr(p, "converted_value_data_type", None),
        })

    # Last assistant turn, even if empty — Moonshot/Kimi sometimes returns HTTP 200
    # with no message body; skipping falsy converted_value hid that and set
    # final_response=None while PyRIT could still report SUCCESS.
    final_response: str | None = None
    for entry in reversed(conversation):
        if entry.get("role") == "assistant":
            val = entry.get("converted_value")
            final_response = "" if val is None else str(val)
            break
    return conversation or None, final_response


async def run_single_cell(
    target: OpenAIChatTarget,
    adversary: OpenAIChatTarget,
    method_name: str,
    category: str,
    prompt: str,
    model_display_name: str,
    adversary_display_name: str | None = None,
    prompt_id: str | None = None,
) -> CellResult:
    """Execute one cell of the test matrix with retry logic."""
    build_fn = ATTACK_METHODS[method_name]
    scorer_id = PLACEHOLDER_SCORER if method_name in PLACEHOLDER_SCORER_METHODS else None
    t0 = time.monotonic()
    last_exc: BaseException | None = None

    memory_labels: dict[str, str] = {
        "model": model_display_name,
        "method": method_name,
        "category": category,
    }
    if prompt_id:
        memory_labels["prompt_id"] = prompt_id

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            attack = build_fn(target, adversary, prompt)
            result = await attack.execute_async(
                objective=prompt,
                memory_labels=memory_labels,
            )
            elapsed = time.monotonic() - t0

            outcome = str(result.outcome)
            logger.info(
                "  [OK] %s | %s | %s — %s (%.1fs)",
                model_display_name, method_name, category, outcome, elapsed,
            )

            conversation, final_response = _extract_transcript(str(result.conversation_id))
            if "SUCCESS" in outcome and not (final_response or "").strip():
                logger.warning(
                    "  [HOLLOW] %s | %s | %s — victim assistant text is empty; "
                    "outcome was %s (placeholder scorer may not reflect usefulness).",
                    model_display_name, method_name, category, outcome,
                )

            return CellResult(
                model=model_display_name,
                method=method_name,
                category=category,
                prompt=prompt,
                status=outcome,
                conversation_id=str(result.conversation_id),
                elapsed_seconds=elapsed,
                prompt_id=prompt_id,
                adversary=adversary_display_name,
                scorer=scorer_id,
                final_response=final_response,
                conversation=conversation,
            )

        except Exception as exc:
            last_exc = exc
            # Refusals / content-filter blocks are deterministic — retrying the
            # same prompt won't change the outcome. Break out of the retry loop
            # immediately so the matrix moves on to the next cell.
            is_safety_now, _ = _classify_safety(exc)
            if is_safety_now:
                logger.warning(
                    "  [SAFETY-EARLY] %s | %s | %s — provider refused; skipping retries.",
                    model_display_name, method_name, category,
                )
                break
            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                logger.warning(
                    "  [RETRY %d/%d] %s | %s | %s — %r. Waiting %ds...",
                    attempt, MAX_RETRIES, model_display_name,
                    method_name, category, exc, backoff,
                )
                await asyncio.sleep(backoff)

    # Retries exhausted — surface full diagnostic context, not a one-line shrug.
    elapsed = time.monotonic() - t0
    exc = last_exc
    is_safety, safety_message = _classify_safety(exc)
    if is_safety:
        logger.warning(
            "  [SAFETY] %s | %s | %s — provider blocked: %s",
            model_display_name, method_name, category, safety_message,
        )
    else:
        logger.error(
            "  [FAIL] %s | %s | %s — %r (exhausted retries, %.1fs)",
            model_display_name, method_name, category, exc, elapsed,
        )
    return CellResult(
        model=model_display_name,
        method=method_name,
        category=category,
        prompt=prompt,
        status=(
            f"CONTENT_FILTER: {safety_message}" if is_safety
            else (f"ERROR: {type(exc).__name__}: {exc}" if exc else "ERROR: unknown")
        ),
        conversation_id=None,
        elapsed_seconds=elapsed,
        prompt_id=prompt_id,
        adversary=adversary_display_name,
        scorer=scorer_id,
        error_type=type(exc).__name__ if exc else None,
        error_message=str(exc) if exc else None,
        traceback=(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if exc else None
        ),
        content_filter=is_safety,
        safety_message=safety_message,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CRESCENDO_PROVIDERS: frozenset[str] = frozenset({"deepseek", "moonshot", "together", "anthropic_opus"})

# Kimi (Moonshot) is the default adversary for PAIR/Crescendo — override with ADVERSARY_PROVIDER.
DEFAULT_ADVERSARY_PROVIDER = "moonshot"


def _is_same_model(a: ModelConfig, b: ModelConfig) -> bool:
    """Return True when two registry rows point to the same logical model."""
    return a.provider == b.provider and a.model_name == b.model_name


def _cell_key(
    model: str,
    method: str,
    category: str,
    prompt_id: str | None,
    prompt: str,
) -> str:
    """Stable identifier for a (model, method, category, prompt) cell.

    Uses ``prompt_id`` when present (rich schema). For legacy prompts without
    an id, falls back to a short SHA-1 of the prompt text so the same prompt
    string keys to the same cell across runs.
    """
    pid = prompt_id or f"hash:{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:12]}"
    return f"{model}||{method}||{category}||{pid}"


def _slug(name: str) -> str:
    """Filesystem-safe slug for a model display name (e.g. "GPT-5.4" -> "gpt_5_4")."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _cell_path(output_root: Path, model_display: str, method: str, filename: str) -> Path:
    """Resolve the per-(model, method) JSONL file inside the structured output tree."""
    return output_root / _slug(model_display) / method / filename


def _serialize_cell(
    cell: "CellResult",
    model_config: ModelConfig,
    factory_name: str,
    mode: str,
) -> dict:
    """Render a CellResult into the on-disk per-cell JSON object."""
    error = None
    if cell.error_type or cell.error_message or cell.content_filter:
        error = {
            "type": cell.error_type,
            "message": cell.error_message,
            "traceback": cell.traceback,
            "content_filter": cell.content_filter,
            "safety_message": cell.safety_message,
        }
    canonical_response = (
        cell.final_response
        if cell.final_response is not None
        else (cell.safety_message or cell.error_message)
    )
    has_scorable_text = bool((canonical_response or "").strip())

    return {
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": asdict(model_config),
        "attack": {"method": cell.method, "factory": factory_name},
        "objective": cell.prompt,
        "category": cell.category,
        "prompt_id": cell.prompt_id,
        "ok": error is None,
        "outcome": cell.status,
        "conversation_id": cell.conversation_id,
        "elapsed_seconds": cell.elapsed_seconds,
        "adversary": cell.adversary,
        "scorer": cell.scorer,
        # Canonical "what the model returned" field for the downstream scorer.
        # Populated in both refusal paths so the scorer has one place to read:
        #   - HTTP 200 with text  -> last assistant turn from the conversation
        #   - HTTP error / 204 / content_filter -> the provider's safety/error string
        "final_response": canonical_response,
        # Scoring inclusion signals:
        #   response_received: True if there is non-empty victim text to judge OR an
        #     explicit API-level refusal (content_filter). Empty 200 bodies count as
        #     False so aggregates don't treat hollow replies as scored responses.
        #   is_refused: True when we *know* the cell is a refusal (API content
        #     filter blocked it). For HTTP 200 responses defaults to False —
        #     StrongREJECT (or whichever judge replaces the placeholder scorer)
        #     overwrites this once it's wired up.
        "response_received": has_scorable_text or bool(cell.content_filter),
        "is_refused": bool(cell.content_filter),
        "conversation": cell.conversation,
        "error": error,
    }


def _load_completed_keys(output_root: Path) -> set[str]:
    """Walk every ``*.jsonl`` under ``output_root`` and collect completed cell keys.

    Malformed lines are warned about and skipped — a partial last line from a
    SIGKILL doesn't poison the resume.
    """
    completed: set[str] = set()
    if not output_root.exists():
        return completed
    for jsonl_path in sorted(output_root.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line %d in %s", line_no, jsonl_path)
                    continue
                model = row.get("model")
                model_name = model.get("display_name", "") if isinstance(model, dict) else (model or "")
                attack = row.get("attack")
                method = attack.get("method", "") if isinstance(attack, dict) else (row.get("method", ""))
                completed.add(
                    _cell_key(
                        model_name,
                        method,
                        row.get("category", ""),
                        row.get("prompt_id"),
                        row.get("objective") or row.get("prompt", ""),
                    )
                )
    return completed


def _extract_prompt(prompt_obj: dict | str, category: str) -> tuple[str, str | None]:
    """Normalize rich/legacy prompt entries into (prompt_text, prompt_id)."""
    if isinstance(prompt_obj, dict):
        prompt_text = prompt_obj.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(
                f"Invalid prompt object in category {category!r}: missing/empty 'prompt_text'"
            )
        prompt_id = prompt_obj.get("prompt_id")
        return prompt_text, prompt_id if isinstance(prompt_id, str) else None

    if not isinstance(prompt_obj, str) or not prompt_obj.strip():
        raise ValueError(f"Invalid legacy prompt in category {category!r}: expected non-empty string")
    return prompt_obj, None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the model x attack-method x prompt matrix."
    )
    parser.add_argument(
        "--prompt-file",
        action="append",
        default=None,
        dest="prompt_file",
        help=(
            "Path to a prompt JSON file (default: prompts/prompts_long.json). "
            "Repeat the flag to pass multiple files to --crescendo-debug; the "
            "standard matrix accepts only one."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Run only the first N prompts across all categories.",
    )
    parser.add_argument(
        "--prompts-per-category",
        type=int,
        default=None,
        help="Keep only the first N prompts within each category (e.g. 1 for smoke tests).",
    )
    parser.add_argument(
        "--method",
        choices=sorted(ATTACK_METHODS.keys()),
        default=None,
        help="Run only this attack method (default: all methods).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Substring filter on model provider OR display name (case-insensitive). "
            "E.g. 'claude' or 'anthropic' for Claude Sonnet 4.6, 'kimi' or 'moonshot' "
            "for Kimi K2.5. Default: every model whose API key is set in .env."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="../results",
        help=(
            "Root directory for results. Each cell is appended to "
            "<root>/<model_slug>/<method>/<N>_<ddmmyy>_<dataset>.jsonl, where N "
            "is the total prompt count, ddmmyy is today's date, and dataset is "
            "the prompt-file stem (e.g. dummy_prompts). An interrupted run can "
            "be resumed by re-invoking the same command — cells already present "
            "anywhere under the root are skipped."
        ),
    )
    parser.add_argument(
        "--mode",
        default="smoke",
        help="Run label written into each JSON record's 'mode' field (default: smoke).",
    )

    # ------------------------------------------------------------------
    # Crescendo-debug subcommand: full multi-turn self-play with transcript
    # logging, defender-refusal classification, and resume-from-JSONL. This
    # is a separate path from --method crescendo (which is one cell of the
    # standard matrix); see the module docstring above for the distinction.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--crescendo-debug",
        action="store_true",
        help=(
            "Run Crescendo (Kimi attacker → DeepSeek defender by default) for JSONL "
            "traces. Without --crescendo-debug-full: one prompt per --prompt-file "
            "(defaults: long + short smoke). With --crescendo-debug-full: every "
            "prompt in each file (default file: BioRT-Bench long, 40 prompts)."
        ),
    )
    parser.add_argument(
        "--crescendo-debug-full",
        action="store_true",
        help=(
            "With --crescendo-debug: load all prompts from each --prompt-file "
            f"(if omitted: see --crescendo-bench, defaulting to long: "
            f"{DEFAULT_FULL_BENCH_PROMPT_FILE}). "
            "CrescendoAttack uses MAX_CRESCENDO_TURNS / MAX_CRESCENDO_BACKTRACKS from "
            "backend/attacks.py."
        ),
    )
    parser.add_argument(
        "--crescendo-bench",
        choices=["long", "short"],
        default="long",
        help=(
            "With --crescendo-debug-full and no --prompt-file: use the full BioRT long (40) "
            f"or short (40) file ({DEFAULT_SHORT_BENCH_PROMPT_FILE}). Default: long. "
            "Ignored if --prompt-file is set or if --crescendo-debug-full is not used."
        ),
    )
    parser.add_argument(
        "--crescendo-moonshot-both",
        action="store_true",
        help=(
            "With --crescendo-debug: Kimi (moonshot) for both Crescendo adversary and "
            "target (two clients); adversarial red-team = attacker in JSONL. "
            "With --crescendo-debug-full and no --prompt-file, add --crescendo-bench short "
            "for the 40-prompt short bench. Writes "
            "crescendo_defender-moonshot_attacker-moonshot_*.jsonl."
        ),
    )
    parser.add_argument(
        "--crescendo-anthropic-both",
        action="store_true",
        help=(
            "With --crescendo-debug: Claude (anthropic) attacks, Kimi (moonshot) defends "
            "(CLI name is legacy). At most one of --crescendo-moonshot-both, "
            "--crescendo-anthropic-both, --crescendo-openai-both. "
            "Output: crescendo_defender-moonshot_attacker-anthropic_*.jsonl."
        ),
    )
    parser.add_argument(
        "--crescendo-openai-both",
        action="store_true",
        dest="crescendo_kimi_vs_openai",
        help=(
            "With --crescendo-debug: Kimi (moonshot) attacks, OpenAI (GPT) defends. "
            "Same as --crescendo-kimi-attacks-openai. "
            "Output: crescendo_defender-openai_attacker-moonshot_*.jsonl."
        ),
    )
    parser.add_argument(
        "--crescendo-kimi-attacks-openai",
        action="store_true",
        dest="crescendo_kimi_vs_openai",
        help=(
            "With --crescendo-debug: Kimi (moonshot) is always the Crescendo adversary; "
            "OpenAI is the target (defender). JSONL: attacker_provider=moonshot, "
            "defender_provider=openai."
        ),
    )
    parser.add_argument(
        "--crescendo-kimi-attacks-anthropic",
        action="store_true",
        help=(
            "With --crescendo-debug: Kimi (moonshot) attacks, Claude (anthropic) defends; "
            "JSONL metadata.attacker_provider is always moonshot. "
            "With --crescendo-debug-full and no --prompt-file, use --crescendo-bench short "
            "to sweep the 40-prompt short bench. "
            "Output: crescendo_defender-anthropic_attacker-moonshot_*.jsonl."
        ),
    )
    parser.add_argument(
        "--defender-provider",
        default=DEFAULT_DEBUG_DEFENDER_PROVIDER,
        help="Defender provider for --crescendo-debug (default: deepseek).",
    )
    parser.add_argument(
        "--attacker-provider",
        default=DEFAULT_DEBUG_ATTACKER_PROVIDER,
        help="Attacker provider for --crescendo-debug (default: moonshot/kimi).",
    )
    parser.add_argument(
        "--debug-output",
        default=DEFAULT_DEBUG_OUTPUT_DIR,
        help=(
            "Crescendo debug output directory or a single .jsonl path. Relative paths "
            "are resolved from the project root (not the shell cwd). One run always "
            "produces a single new .jsonl (default: <repo>/results/crescendo/...jsonl; "
            "a path ending in .jsonl writes all lines to that file only)."
        ),
    )
    parser.add_argument(
        "--crescendo-resume-from-jsonl",
        metavar="PATH",
        default=None,
        help=(
            "Path to a partial Crescendo .jsonl from a stopped run. Skips any "
            "metadata.prompt_id already present, runs only the rest, and should be "
            "used with the same --prompt-file / --crescendo-bench as the original. "
            "To append to the same artifact, set --debug-output to that exact .jsonl "
            "file (otherwise a new timestamped file is created under a directory)."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Crescendo-debug flag validation (port of root /matrix_runner.py).
    # ------------------------------------------------------------------
    if args.crescendo_debug_full and not args.crescendo_debug:
        parser.error("--crescendo-debug-full requires --crescendo-debug")
    if args.crescendo_resume_from_jsonl and not args.crescendo_debug:
        parser.error("--crescendo-resume-from-jsonl requires --crescendo-debug")
    for _flag, _name in (
        (args.crescendo_moonshot_both, "--crescendo-moonshot-both"),
        (args.crescendo_anthropic_both, "--crescendo-anthropic-both"),
        (args.crescendo_kimi_vs_openai, "--crescendo-openai-both or --crescendo-kimi-attacks-openai"),
        (args.crescendo_kimi_attacks_anthropic, "--crescendo-kimi-attacks-anthropic"),
    ):
        if _flag and not args.crescendo_debug:
            parser.error("%s requires --crescendo-debug" % _name)
    _selfplay_count = (
        int(bool(args.crescendo_moonshot_both))
        + int(bool(args.crescendo_anthropic_both))
        + int(bool(args.crescendo_kimi_vs_openai))
        + int(bool(args.crescendo_kimi_attacks_anthropic))
    )
    if _selfplay_count > 1:
        parser.error(
            "use at most one of --crescendo-moonshot-both, "
            "--crescendo-anthropic-both, --crescendo-openai-both / "
            "--crescendo-kimi-attacks-openai, --crescendo-kimi-attacks-anthropic"
        )

    # ------------------------------------------------------------------
    # Crescendo-debug dispatch — short-circuits the standard matrix loop.
    # ------------------------------------------------------------------
    if args.crescendo_debug:
        if args.prompt_file:
            prompt_files = args.prompt_file
        elif args.crescendo_debug_full:
            prompt_files = (
                [DEFAULT_SHORT_BENCH_PROMPT_FILE]
                if args.crescendo_bench == "short"
                else [DEFAULT_FULL_BENCH_PROMPT_FILE]
            )
        elif args.crescendo_bench == "short":
            prompt_files = [DEFAULT_SHORT_BENCH_PROMPT_FILE]
        else:
            prompt_files = list(DEFAULT_DEBUG_PROMPT_FILES)

        resume_skip: set[str] | None = None
        if args.crescendo_resume_from_jsonl:
            resume_skip = load_completed_prompt_ids_from_jsonl(
                args.crescendo_resume_from_jsonl
            )

        if args.crescendo_moonshot_both:
            await run_crescendo_moonshot_as_both_roles(
                prompt_files=prompt_files,
                run_single_cell_fn=run_single_cell,
                output_path=args.debug_output,
                load_all_prompts=bool(args.crescendo_debug_full),
                skip_prompt_ids=resume_skip,
            )
        elif args.crescendo_anthropic_both:
            await run_crescendo_anthropic_as_both_roles(
                prompt_files=prompt_files,
                run_single_cell_fn=run_single_cell,
                output_path=args.debug_output,
                load_all_prompts=bool(args.crescendo_debug_full),
                skip_prompt_ids=resume_skip,
            )
        elif args.crescendo_kimi_vs_openai:
            await run_crescendo_openai_as_both_roles(
                prompt_files=prompt_files,
                run_single_cell_fn=run_single_cell,
                output_path=args.debug_output,
                load_all_prompts=bool(args.crescendo_debug_full),
                skip_prompt_ids=resume_skip,
            )
        elif args.crescendo_kimi_attacks_anthropic:
            await run_crescendo_kimi_attacks_anthropic(
                prompt_files=prompt_files,
                run_single_cell_fn=run_single_cell,
                output_path=args.debug_output,
                load_all_prompts=bool(args.crescendo_debug_full),
                skip_prompt_ids=resume_skip,
            )
        else:
            await run_crescendo_debug_smoke_test(
                prompt_files=prompt_files,
                run_single_cell_fn=run_single_cell,
                attacker_provider=args.attacker_provider,
                defender_provider=args.defender_provider,
                output_path=args.debug_output,
                load_all_prompts=bool(args.crescendo_debug_full),
                skip_prompt_ids=resume_skip,
            )
        return

    # ------------------------------------------------------------------
    # Standard matrix path — exactly one prompt file allowed.
    # ------------------------------------------------------------------
    if args.prompt_file and len(args.prompt_file) > 1:
        parser.error(
            "--prompt-file may only be specified once outside --crescendo-debug "
            "(got %d files)." % len(args.prompt_file)
        )
    prompt_file_single = (args.prompt_file or ["prompts/prompts_long.json"])[0]

    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    prompts_by_category = limit_per_category(
        load_prompts(prompt_file_single),
        args.prompts_per_category,
    )
    prompts_by_category = limit_prompts(prompts_by_category, args.max_prompts)

    # Filter the attack method dict if --method was given. Use the original
    # ATTACK_METHODS for choices validation (above); here we just narrow.
    selected_methods: dict[str, object] = (
        {args.method: ATTACK_METHODS[args.method]} if args.method else dict(ATTACK_METHODS)
    )

    available_models = get_available_models()

    # Pre-build victim targets with the response cap. Models whose API key is
    # missing have already been filtered by get_available_models; here we
    # additionally skip providers whose target can't yet be built (e.g. a
    # provider whose adapter raises NotImplementedError).
    buildable: list[tuple[ModelConfig, OpenAIChatTarget]] = []
    for cfg in available_models:
        try:
            victim_cap = (
                cfg.victim_max_tokens
                if cfg.victim_max_tokens is not None
                else MAX_TARGET_COMPLETION_TOKENS
            )
            buildable.append((cfg, build_target(cfg, max_tokens=victim_cap)))
        except NotImplementedError as exc:
            logger.warning("Skipping %s: %s", cfg.display_name, exc)

    if not buildable:
        raise RuntimeError("No models could be built. Check your .env and config/models.py.")

    # Adversary selection: ADVERSARY_PROVIDER env var, else Kimi (moonshot).
    # The adversary config is chosen from buildable (so we honor available
    # API keys) but the actual adversary target is rebuilt without the
    # response cap — PAIR/Crescendo need headroom for jailbreak generation.
    raw_adv = os.getenv("ADVERSARY_PROVIDER", DEFAULT_ADVERSARY_PROVIDER)
    adversary_provider = (raw_adv or DEFAULT_ADVERSARY_PROVIDER).strip()
    adversary_config: ModelConfig | None = None
    for cfg, _ in buildable:
        if cfg.provider == adversary_provider:
            adversary_config = cfg
            break
    if adversary_config is None:
        raise RuntimeError(
            f"ADVERSARY_PROVIDER={adversary_provider!r} but no buildable model "
            f"has that provider. Set MOONSHOT_API_KEY (or the matching key) and re-run. "
            f"Available: {[c.provider for c, _ in buildable]}"
        )
    adversary_target: OpenAIChatTarget = build_target(adversary_config)

    logger.info(
        "Adversary / scorer LLM: %s (uncapped) | victim cap: %d tokens",
        adversary_config.display_name, MAX_TARGET_COMPLETION_TOKENS,
    )

    # --model filter applied AFTER adversary selection so the adversary target
    # stays valid even when the user narrows the matrix to a single victim.
    if args.model:
        needle = args.model.lower()
        filtered = [
            (cfg, tgt) for cfg, tgt in buildable
            if needle in cfg.provider.lower() or needle in cfg.display_name.lower()
        ]
        if not filtered:
            raise RuntimeError(
                f"--model {args.model!r} matched no buildable models. "
                f"Available: {[c.display_name for c, _ in buildable]}"
            )
        logger.info(
            "--model %r narrowed matrix to: %s",
            args.model, [c.display_name for c, _ in filtered],
        )
        buildable = filtered

    prompt_count = sum(len(p) for p in prompts_by_category.values())
    total_cells_upper = len(buildable) * len(selected_methods) * prompt_count
    total_cells_expected = 0
    for model_config, _ in buildable:
        for method_name in selected_methods:
            if (
                method_name in METHODS_REQUIRING_ADVERSARY
                and _is_same_model(adversary_config, model_config)
            ):
                continue
            if (
                method_name == "crescendo"
                and model_config.provider not in CRESCENDO_PROVIDERS
            ):
                continue
            total_cells_expected += prompt_count

    logger.info(
        "Matrix: %d models x %d methods x %d prompts = %d cells (expected run) / %d upper bound",
        len(buildable), len(selected_methods), prompt_count, total_cells_expected, total_cells_upper,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Filename: <prompt_count>_<ddmmyy>_<dataset>.jsonl. Same prompt count +
    # same day + same dataset appends to the same file, which composes
    # naturally with resume logic.
    dataset_name = Path(prompt_file_single).stem
    output_filename = (
        f"{prompt_count}_{datetime.now().strftime('%d%m%y')}_{dataset_name}.jsonl"
    )
    completed_keys = _load_completed_keys(output_root)
    if completed_keys:
        logger.info(
            "Resume: %d cell(s) already under %s — will skip them.",
            len(completed_keys), output_root,
        )

    all_results: list[CellResult] = []
    completed = 0
    skipped_resume = 0
    # One file handle per (model, method); opened lazily on first write.
    out_handles: dict[tuple[str, str], TextIO] = {}

    for model_config, target in buildable:
        for method_name in selected_methods:
            # Guard: PAIR / Crescendo are scientifically invalid when target == adversary.
            if (
                method_name in METHODS_REQUIRING_ADVERSARY
                and _is_same_model(adversary_config, model_config)
            ):
                logger.error(
                    "Skipping %s on %s — %s requires a separate adversary.",
                    method_name, model_config.display_name, method_name,
                )
                continue

            # Budget gate: Crescendo is only run against cheap providers.
            if (
                method_name == "crescendo"
                and model_config.provider not in CRESCENDO_PROVIDERS
            ):
                logger.info(
                    "Skipping crescendo on %s (budget gate — provider %s not in %s)",
                    model_config.display_name, model_config.provider,
                    sorted(CRESCENDO_PROVIDERS),
                )
                continue

            for category, category_prompts in prompts_by_category.items():
                for prompt_obj in category_prompts:
                    prompt_text, prompt_id = _extract_prompt(prompt_obj, category)

                    completed += 1
                    cell_key = _cell_key(
                        model_config.display_name, method_name, category,
                        prompt_id, prompt_text,
                    )
                    if cell_key in completed_keys:
                        skipped_resume += 1
                        logger.info(
                            "[%d/%d] SKIP (resumed) %s on %s for %s (%s)",
                            completed, total_cells_expected, method_name,
                            model_config.display_name, category,
                            prompt_id or "no-id",
                        )
                        continue

                    logger.info(
                        "[%d/%d] Running %s on %s for %s (%s)...",
                        completed, total_cells_expected, method_name,
                        model_config.display_name, category,
                        prompt_id or "no-id",
                    )

                    cell_result = await run_single_cell(
                        target=target,
                        adversary=adversary_target,
                        method_name=method_name,
                        category=category,
                        prompt=prompt_text,
                        model_display_name=model_config.display_name,
                        adversary_display_name=adversary_config.display_name,
                        prompt_id=prompt_id,
                    )
                    all_results.append(cell_result)

                    handle_key = (model_config.display_name, method_name)
                    if handle_key not in out_handles:
                        cell_path = _cell_path(
                            output_root, model_config.display_name, method_name, output_filename,
                        )
                        cell_path.parent.mkdir(parents=True, exist_ok=True)
                        out_handles[handle_key] = cell_path.open("a", encoding="utf-8")
                    fh = out_handles[handle_key]
                    factory_name = ATTACK_METHODS[method_name].__name__
                    record = _serialize_cell(
                        cell_result, model_config, factory_name, args.mode,
                    )
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    completed_keys.add(cell_key)

    for fh in out_handles.values():
        fh.close()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("MATRIX RUN COMPLETE")
    print("=" * 70)
    print(f"Results written under: {output_root.resolve()}")
    print(f"Mode: {args.mode}  |  Layout: <model>/<method>/{output_filename}")
    if skipped_resume:
        print(f"Skipped (already on disk): {skipped_resume}")

    safety_blocks = sum(1 for r in all_results if r.content_filter)
    failures = sum(
        1 for r in all_results
        if r.status.startswith("ERROR") or r.status.startswith("CONTENT_FILTER")
    )
    successes = len(all_results) - failures
    print(
        f"Total cells: {len(all_results)}  |  OK: {successes}  |  "
        f"Failed: {failures}  |  Content-filter: {safety_blocks}"
    )

    for result in all_results:
        if result.content_filter:
            status_icon = "!"
        elif result.status.startswith("ERROR"):
            status_icon = "X"
        else:
            status_icon = "+"
        tag = f" [{result.prompt_id}]" if result.prompt_id else ""
        print(
            f"  [{status_icon}] {result.model:<22} | {result.method:<12} "
            f"| {result.category:<16} | {result.status}{tag}"
        )

    # HOOK: StrongREJECT scoring goes here.
    # After the matrix run, iterate over all_results, load each conversation
    # from CentralMemory, and score the final assistant response with StrongREJECT.

    logged_conversations = [r for r in all_results if r.conversation_id]
    print(f"\nConversations stored in memory: {len(logged_conversations)}")
    print("Memory verification complete.")


if __name__ == "__main__":
    asyncio.run(main())
