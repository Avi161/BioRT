"""Hello World milestone — verify single model + memory logging end-to-end.

Sends 5 mock bio-misuse prompts to DeepSeek V4 Flash via a direct request
(no-attack baseline), logs interactions to PyRIT's memory layer, and prints
the stored conversations to verify persistence.

Usage:
    python hello_world.py
"""

from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv

load_dotenv()

from pyrit.executor.attack import (
    AttackExecutor,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.memory import CentralMemory
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

from config.models import build_target, get_available_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_mock_prompts(path: str = "prompts/mock_prompts.json", limit: int = 5) -> list[str]:
    """Load mock prompts from JSON, flattening across categories.

    Args:
        path: Path to the mock prompts JSON file.
        limit: Maximum number of prompts to return.

    Returns:
        A flat list of prompt strings.
    """
    with open(path) as fh:
        data: dict[str, list[str]] = json.load(fh)
    flat = [prompt for prompts in data.values() for prompt in prompts]
    return flat[:limit]


async def main() -> None:
    # PyRIT's IN_MEMORY backend uses SQLite under the hood.
    # All interactions are stored for the session and queryable via CentralMemory.
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    available = get_available_models()
    first_buildable = None
    for cfg in available:
        try:
            first_buildable = (cfg, build_target(cfg))
            break
        except NotImplementedError:
            continue

    if first_buildable is None:
        raise RuntimeError("No models could be built. Check your .env and config/models.py.")

    model_config, target = first_buildable

    prompts = load_mock_prompts()
    logger.info(
        "Sending %d prompts to %s (direct request)...",
        len(prompts), model_config.display_name,
    )

    attack = PromptSendingAttack(objective_target=target)

    results = await AttackExecutor().execute_attack_async(
        attack=attack,
        objectives=prompts,
    )

    printer = ConsoleAttackResultPrinter()
    for result in results:
        await printer.print_result_async(result=result)

    # --- Verify memory logging ---
    memory = CentralMemory.get_memory_instance()
    print("\n" + "=" * 70)
    print("MEMORY VERIFICATION — stored conversations")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        pieces = memory.get_message_pieces(conversation_id=result.conversation_id)
        print(f"\n--- Prompt {i} (conversation {result.conversation_id[:8]}...) ---")
        for piece in pieces:
            role_tag = getattr(piece, "role", None) or getattr(piece, "_role", "unknown")
            value_preview = piece.original_value[:200]
            print(f"  [{str(role_tag).upper()}] {value_preview}")

    print(f"\nTotal conversations logged: {len(results)}")
    print("Memory verification complete.")


if __name__ == "__main__":
    asyncio.run(main())
