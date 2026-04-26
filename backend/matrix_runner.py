"""Matrix Experiment Runner — iterate Model x Method x Category x Prompt.

Orchestrates the full test matrix across all available models and attack
methods, logging every interaction to PyRIT's memory layer.

Usage:
    python matrix_runner.py

The runner skips models whose API keys are missing and continues on
individual prompt failures so the full matrix doesn't crash halfway through.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from pyrit.memory import CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

from attacks import (
    ATTACK_METHODS,
    METHODS_REQUIRING_ADVERSARY,
    PLACEHOLDER_SCORER,
    PLACEHOLDER_SCORER_METHODS,
)
from config.models import ModelConfig, build_target, get_available_models

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


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(
    path: str = "prompts/prompts_long.json",
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
            )

        except Exception as exc:
            last_exc = exc
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
    logger.error(
        "  [FAIL] %s | %s | %s — %r (exhausted retries, %.1fs)",
        model_display_name, method_name, category, exc, elapsed,
    )
    return CellResult(
        model=model_display_name,
        method=method_name,
        category=category,
        prompt=prompt,
        status=f"ERROR: {type(exc).__name__}: {exc}" if exc else "ERROR: unknown",
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
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CRESCENDO_PROVIDERS: frozenset[str] = frozenset({"deepseek", "moonshot", "together"})

# Kimi (Moonshot) is the default adversary for PAIR/Crescendo — override with ADVERSARY_PROVIDER.
DEFAULT_ADVERSARY_PROVIDER = "moonshot"


def _is_same_model(a: ModelConfig, b: ModelConfig) -> bool:
    """Return True when two registry rows point to the same logical model."""
    return a.provider == b.provider and a.model_name == b.model_name


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
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    prompts_by_category = load_prompts()
    available_models = get_available_models()

    # Pre-build targets, filtering out models that can't be constructed yet (e.g. Anthropic).
    buildable: list[tuple[ModelConfig, OpenAIChatTarget]] = []
    for cfg in available_models:
        try:
            buildable.append((cfg, build_target(cfg)))
        except NotImplementedError as exc:
            logger.warning("Skipping %s: %s", cfg.display_name, exc)

    if not buildable:
        raise RuntimeError("No models could be built. Check your .env and config/models.py.")

    # Adversary selection: ADVERSARY_PROVIDER env var, else Kimi (moonshot).
    raw_adv = os.getenv("ADVERSARY_PROVIDER", DEFAULT_ADVERSARY_PROVIDER)
    adversary_provider = (raw_adv or DEFAULT_ADVERSARY_PROVIDER).strip()
    adversary_config: ModelConfig | None = None
    adversary_target: OpenAIChatTarget | None = None

    for cfg, tgt in buildable:
        if cfg.provider == adversary_provider:
            adversary_config, adversary_target = cfg, tgt
            break
    if adversary_config is None:
        raise RuntimeError(
            f"ADVERSARY_PROVIDER={adversary_provider!r} but no buildable model "
            f"has that provider. Set MOONSHOT_API_KEY (or the matching key) and re-run. "
            f"Available: {[c.provider for c, _ in buildable]}"
        )

    logger.info("Adversary / scorer LLM: %s", adversary_config.display_name)

    prompt_count = sum(len(p) for p in prompts_by_category.values())
    total_cells_upper = len(buildable) * len(ATTACK_METHODS) * prompt_count
    total_cells_expected = 0
    for model_config, _ in buildable:
        for method_name in ATTACK_METHODS:
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
        len(buildable), len(ATTACK_METHODS), prompt_count, total_cells_expected, total_cells_upper,
    )

    all_results: list[CellResult] = []
    completed = 0

    for model_config, target in buildable:
        for method_name in ATTACK_METHODS:
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

    # --- Summary ---
    print("\n" + "=" * 70)
    print("MATRIX RUN COMPLETE")
    print("=" * 70)

    successes = sum(1 for r in all_results if not r.status.startswith("ERROR"))
    failures = len(all_results) - successes
    print(f"Total cells: {len(all_results)}  |  OK: {successes}  |  Failed: {failures}")

    for result in all_results:
        status_icon = "+" if not result.status.startswith("ERROR") else "X"
        tag = f" [{result.prompt_id}]" if result.prompt_id else ""
        print(
            f"  [{status_icon}] {result.model:<22} | {result.method:<12} "
            f"| {result.category:<16} | {result.status}{tag}"
        )

    # HOOK: StrongREJECT scoring goes here.
    # After the matrix run, iterate over all_results, load each conversation
    # from CentralMemory, and score the final assistant response with StrongREJECT.

    memory = CentralMemory.get_memory_instance()
    logged_conversations = [r for r in all_results if r.conversation_id]
    print(f"\nConversations stored in memory: {len(logged_conversations)}")
    print("Memory verification complete.")


if __name__ == "__main__":
    asyncio.run(main())
