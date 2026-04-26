"""End-to-end smoke test for the four attack factories in ``attacks.py``.

For each ``(provider, method)`` entry in ``SMOKE_PLAN``, builds the attack
against the configured smoke target, executes it on a single benign
objective ("What is 2+2?"), and writes the full conversation to a JSON
file for human inspection.

Output layout: ``results/{provider}/{method}/{mode}_{iso-timestamp}.json``.

Usage:
    python validate_attacks.py

Exits non-zero if any planned attack fails to complete OR if any
configuration invariant is violated (unknown provider in SMOKE_PLAN,
PAIR/Crescendo planned without a separate adversary, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from pyrit.memory import CentralMemory
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

from attacks import (
    ATTACK_METHODS,
    METHODS_REQUIRING_ADVERSARY,
    PLACEHOLDER_SCORER,
    PLACEHOLDER_SCORER_METHODS,
)
from config.models import MODEL_REGISTRY, ModelConfig, build_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_attacks")

BENIGN_OBJECTIVE = "What is 2+2?"
RESULTS_DIR = Path(__file__).parent / "results"

# Per-attack smoke target. All four attacks run against Kimi (cheap, all attack
# types fit). Bonus: direct call against DeepSeek to verify the adversary LLM
# endpoint is wired up correctly.
SMOKE_PLAN: list[tuple[str, str]] = [
    ("moonshot", "direct"),
    ("moonshot", "base64"),
    ("moonshot", "pair"),
    ("moonshot", "crescendo"),
    ("deepseek", "direct"),
]
# Pre-call sleep (seconds) by provider — used to respect tight per-key limits.
PER_PROVIDER_SLEEP: dict[str, float] = {}


_MISSING = object()


def _outcome_is_success(outcome: Any) -> bool:
    """Return True only for explicit SUCCESS outcomes."""
    name = str(getattr(outcome, "name", "")).upper()
    if name:
        return name == "SUCCESS"
    return str(outcome).strip().upper() in {"ATTACKOUTCOME.SUCCESS", "SUCCESS"}


def _find_config(provider: str) -> ModelConfig | None:
    return next((c for c in MODEL_REGISTRY if c.provider == provider), None)


def _model_metadata(cfg: ModelConfig | None) -> dict[str, Any] | None:
    """Static model metadata for the JSON record. Strips api_key_env so the env
    var name never leaks into stored payloads."""
    if cfg is None:
        return None
    md = asdict(cfg)
    md.pop("api_key_env", None)
    return md


def _serialize_pieces(conversation_id: Any) -> list[dict[str, Any]]:
    """Pull message pieces from PyRIT memory and convert to plain dicts.

    PyRIT v0.13's ``MessagePiece`` exposes role via ``api_role`` (public) and
    stores the constructor argument under ``_role``. We try them in that order
    and **fail loudly** if neither is present — a silent default would mask a
    real schema drift in a future PyRIT release.
    """
    memory = CentralMemory.get_memory_instance()
    pieces = memory.get_message_pieces(conversation_id=conversation_id)
    out: list[dict[str, Any]] = []
    for p in pieces:
        role = getattr(p, "api_role", _MISSING)
        if role is _MISSING:
            role = getattr(p, "_role", _MISSING)
        if role is _MISSING:
            raise RuntimeError(
                f"PyRIT message piece exposes neither `api_role` nor `_role` "
                f"(type={type(p).__name__}). Schema drift — update _serialize_pieces."
            )
        original_value = getattr(p, "original_value", _MISSING)
        if original_value is _MISSING:
            raise RuntimeError(
                f"PyRIT message piece is missing `original_value` "
                f"(type={type(p).__name__}). Schema drift — update _serialize_pieces."
            )
        out.append({
            "role": str(role),
            "original_value": original_value,
            "converted_value": getattr(p, "converted_value", None),
            "original_value_data_type": getattr(p, "original_value_data_type", None),
            "converted_value_data_type": getattr(p, "converted_value_data_type", None),
        })
    return out


def _write_result_json(record: dict[str, Any]) -> Path:
    """Write a result record to results/{provider}/{method}/{mode}_{ts}.json.

    Timestamp is microsecond-precision so two attacks completing in the
    same UTC second do not collide and silently overwrite earlier output.
    """
    ts = record["timestamp_utc"].replace(":", "").replace("-", "").replace(".", "")
    provider = record["model"]["provider"]
    method = record["attack"]["method"]
    out_dir = RESULTS_DIR / provider / method
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{record['mode']}_{ts}.json"
    path = out_dir / filename
    with path.open("w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return path


def _display_path(path: Path) -> str:
    """Best-effort relative path for log lines; falls back to absolute if cwd
    is not an ancestor of the results directory."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


async def _run_attack(
    method_name: str,
    build_fn,
    target,
    adversary,
    target_cfg: ModelConfig,
    adversary_cfg: ModelConfig | None,
    mode: str,
) -> tuple[bool, str, Path]:
    """Run one attack, persist its result, return (executed_without_error, summary, path).

    The first tuple element is purely "did the harness run end-to-end without
    raising?"; whether the *attack* succeeded against the target's safeguards is
    recorded separately as ``record["attack_succeeded"]`` from ``result.outcome``.
    """
    # Microsecond precision so two same-second attacks do not collide on disk.
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    scorer_id = PLACEHOLDER_SCORER if method_name in PLACEHOLDER_SCORER_METHODS else None
    record: dict[str, Any] = {
        "mode": mode,  # "smoke" | "real"
        "timestamp_utc": timestamp,
        "model": _model_metadata(target_cfg),
        "adversary": _model_metadata(adversary_cfg),
        "attack": {
            "method": method_name,
            "factory": getattr(build_fn, "__name__", str(build_fn)),
            "scorer": scorer_id,
        },
        "objective": BENIGN_OBJECTIVE,
        "executed_without_error": False,
        "attack_succeeded": None,
        "outcome": None,
        "conversation_id": None,
        "elapsed_seconds": None,
        "conversation": [],
        "error": None,
    }

    t0 = time.monotonic()
    try:
        attack = build_fn(target, adversary, BENIGN_OBJECTIVE)
        result = await attack.execute_async(
            objective=BENIGN_OBJECTIVE,
            memory_labels={"validation": mode, "method": method_name},
        )
        elapsed = time.monotonic() - t0

        record["elapsed_seconds"] = round(elapsed, 3)
        record["conversation_id"] = str(result.conversation_id)
        outcome_str = str(result.outcome)
        record["outcome"] = outcome_str
        record["conversation"] = _serialize_pieces(result.conversation_id)
        record["executed_without_error"] = bool(record["conversation"])
        record["attack_succeeded"] = _outcome_is_success(result.outcome)

        path = _write_result_json(record)
        summary = (
            f"{method_name}: outcome={result.outcome} "
            f"conversation_id={record['conversation_id'][:8]}... "
            f"pieces={len(record['conversation'])} elapsed={elapsed:.1f}s "
            f"-> {_display_path(path)}"
        )
        return record["executed_without_error"], summary, path

    except Exception as exc:
        record["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
        path = _write_result_json(record)
        summary = (
            f"{method_name}: FAILED — {type(exc).__name__}: {exc} "
            f"(see traceback in {_display_path(path)})"
        )
        return False, summary, path


def _validate_smoke_plan() -> None:
    """Catch SMOKE_PLAN typos at startup before any API calls happen.

    A typo like ``("googel", "direct")`` would otherwise surface as a single
    WARNING line and silently never exercise the ``direct`` attack.
    """
    known_providers = {c.provider for c in MODEL_REGISTRY}
    known_methods = set(ATTACK_METHODS.keys())
    bad: list[str] = []
    for provider, method_name in SMOKE_PLAN:
        if provider not in known_providers:
            bad.append(f"unknown provider {provider!r} (known: {sorted(known_providers)})")
        if method_name not in known_methods:
            bad.append(f"unknown method {method_name!r} (known: {sorted(known_methods)})")
    if bad:
        raise RuntimeError("Invalid SMOKE_PLAN:\n  - " + "\n  - ".join(bad))


async def main() -> int:
    _validate_smoke_plan()

    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    # Adversary / scorer LLM: honour ADVERSARY_PROVIDER env var, default Kimi (moonshot).
    adversary_provider = (os.getenv("ADVERSARY_PROVIDER", "moonshot") or "moonshot").strip()
    adversary_cfg_resolved = _find_config(adversary_provider)
    adversary: Any = None
    if adversary_cfg_resolved is None:
        logger.warning(
            "No %r provider in MODEL_REGISTRY — adversary unavailable.",
            adversary_provider,
        )
    else:
        try:
            adversary = build_target(adversary_cfg_resolved)
            logger.info("Adversary / scorer LLM: %s", adversary_cfg_resolved.display_name)
        except EnvironmentError as exc:
            logger.error("Adversary LLM unavailable (missing key): %s", exc)
        except Exception as exc:  # noqa: BLE001 — log and continue with adversary=None
            logger.exception("Adversary LLM build failed (config error): %r", exc)

    # Hard-fail if the plan needs an adversary but none is available — running
    # PAIR/Crescendo target-vs-target is scientifically invalid.
    if adversary is None:
        needs_adversary = [
            (p, m) for p, m in SMOKE_PLAN if m in METHODS_REQUIRING_ADVERSARY
        ]
        if needs_adversary:
            logger.error(
                "SMOKE_PLAN requests %s but no adversary LLM is buildable. "
                "Set %s or remove these entries from SMOKE_PLAN.",
                needs_adversary,
                adversary_cfg_resolved.api_key_env if adversary_cfg_resolved else "ADVERSARY_PROVIDER",
            )
            return 2

    target_cache: dict[str, tuple[ModelConfig, Any]] = {}

    def _get(provider: str) -> tuple[ModelConfig, Any]:
        if provider in target_cache:
            return target_cache[provider]
        cfg = _find_config(provider)
        # _validate_smoke_plan already verified provider is in registry.
        assert cfg is not None, f"Provider {provider!r} unexpectedly missing"
        t = build_target(cfg)
        target_cache[provider] = (cfg, t)
        return cfg, t

    results: list[tuple[bool, str, Path]] = []
    for provider, method_name in SMOKE_PLAN:
        try:
            cfg, target = _get(provider)
        except EnvironmentError as exc:
            logger.warning("Skipping %s/%s — missing key: %s", provider, method_name, exc)
            continue

        sleep_s = PER_PROVIDER_SLEEP.get(provider, 0.0)
        if sleep_s:
            logger.info("Sleeping %.0fs before %s call (rate limit)", sleep_s, provider)
            await asyncio.sleep(sleep_s)

        # adversary is guaranteed non-None for METHODS_REQUIRING_ADVERSARY at
        # this point (hard-fail above). For direct/base64, falling back to the
        # target as a no-op "adversary" is fine — they don't read it.
        adv = adversary if adversary is not None else target
        adv_cfg = adversary_cfg_resolved if adversary is not None else None
        logger.info("→ %s on %s", method_name, cfg.display_name)
        ok, summary, path = await _run_attack(
            method_name, ATTACK_METHODS[method_name],
            target, adv, cfg, adv_cfg, mode="smoke",
        )
        logger.info("  %s", summary)
        results.append((ok, summary, path))

    if not results:
        logger.error(
            "No smoke tests executed. Ensure at least one provider key is set for SMOKE_PLAN."
        )
        return 2

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY — smoke (per-attack target mapping)")
    print("=" * 70)
    for ok, summary, _ in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {summary}")

    n_failed = sum(1 for ok, _, _ in results if not ok)
    print(f"\n{len(results) - n_failed}/{len(results)} passed")
    print(f"Result JSONs in: {RESULTS_DIR}/")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
