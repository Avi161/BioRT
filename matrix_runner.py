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

def load_prompts(path: str = "prompts/mock_prompts.json") -> dict[str, list[str]]:
    """Load categorized prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        Dict mapping category names to lists of prompt strings.
    """
    with open(path) as fh:
        return json.load(fh)


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
) -> CellResult:
    """Execute one cell of the test matrix with retry logic."""
    build_fn = ATTACK_METHODS[method_name]
    scorer_id = PLACEHOLDER_SCORER if method_name in PLACEHOLDER_SCORER_METHODS else None
    t0 = time.monotonic()
    last_exc: BaseException | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            attack = build_fn(target, adversary, prompt)
            result = await attack.execute_async(
                objective=prompt,
                memory_labels={
                    "model": model_display_name,
                    "method": method_name,
                    "category": category,
                },
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

    # Use the first buildable model as the adversary / scorer LLM.
    adversary_config, adversary_target = buildable[0]
    logger.info("Adversary / scorer LLM: %s", adversary_config.display_name)

    prompt_count = sum(len(p) for p in prompts_by_category.values())
    total_cells = len(buildable) * len(ATTACK_METHODS) * prompt_count
    logger.info(
        "Matrix: %d models x %d methods x %d prompts = %d cells",
        len(buildable), len(ATTACK_METHODS), prompt_count, total_cells,
    )

    all_results: list[CellResult] = []
    completed = 0

    for model_config, target in buildable:
        for method_name in ATTACK_METHODS:
            # Guard: PAIR / Crescendo are scientifically invalid when target == adversary
            # (target attacks itself, scores itself). Skip with a loud error rather than
            # silently produce results that look fine but mean nothing.
            if (
                method_name in METHODS_REQUIRING_ADVERSARY
                and adversary_config.display_name == model_config.display_name
            ):
                logger.error(
                    "Skipping %s on %s — only one buildable model, %s requires a separate "
                    "adversary. Add another API key and re-run.",
                    method_name, model_config.display_name, method_name,
                )
                continue
            for category, category_prompts in prompts_by_category.items():
                for prompt in category_prompts:
                    completed += 1
                    logger.info(
                        "[%d/%d] Running %s on %s for %s...",
                        completed, total_cells, method_name,
                        model_config.display_name, category,
                    )

                    cell_result = await run_single_cell(
                        target=target,
                        adversary=adversary_target,
                        method_name=method_name,
                        category=category,
                        prompt=prompt,
                        model_display_name=model_config.display_name,
                        adversary_display_name=adversary_config.display_name,
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
        print(
            f"  [{status_icon}] {result.model:<22} | {result.method:<12} "
            f"| {result.category:<16} | {result.status}"
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
