"""Model registry and PyRIT target factory.

Centralizes model configurations so hello_world.py, matrix_runner.py, and
validate_attacks.py share the same definitions. Every supported provider
exposes (or can be wrapped into) an OpenAI-compatible chat/completions
endpoint, so a single OpenAIChatTarget class works for all of them.

Anthropic's OpenAI-compatible base URL (https://api.anthropic.com/v1/) is
used for Claude — see Anthropic's "OpenAI SDK compatibility" documentation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from pyrit.prompt_target import OpenAIChatTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    provider: str
    endpoint: str
    api_key_env: str
    model_name: str
    # Optional kwargs forwarded to OpenAIChatTarget. Set temperature=0.0 on
    # adversary/scorer LLMs that ship "thinking" tokens (DeepSeek V4) to
    # suppress chain-of-thought output that inflates cost and confuses parsers.
    temperature: float | None = None
    extra_body: dict[str, Any] | None = field(default=None)
    # Role hints — informational only, used by the report writer to label rows.
    # "frontier_closed" | "frontier_supplementary" | "open_weight_control"
    role: str = "frontier_closed"


MODEL_REGISTRY: list[ModelConfig] = [
    # Note: matrix_runner.py chooses adversary by ADVERSARY_PROVIDER (default moonshot),
    # not by registry order. DeepSeek remains near the top as a cheap defender row.
    ModelConfig(
        display_name="DeepSeek V4 Flash",
        provider="deepseek",
        endpoint="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model_name="deepseek-chat",
        temperature=0.0,
        role="frontier_supplementary",
    ),
    ModelConfig(
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        # Anthropic exposes an OpenAI-SDK-compatible base URL; OpenAIChatTarget works.
        endpoint="https://api.anthropic.com/v1/",
        api_key_env="ANTHROPIC_API_KEY",
        model_name="claude-sonnet-4-6",
        role="frontier_closed",
    ),
    ModelConfig(
        display_name="GPT-5.4",
        provider="openai",
        endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_name="gpt-5.4",
        role="frontier_closed",
    ),
    ModelConfig(
        display_name="Gemini 3 Pro",
        provider="google",
        endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        model_name="gemini-3-pro-preview",
        role="frontier_closed",
    ),
    ModelConfig(
        display_name="Grok 4",
        provider="xai",
        endpoint="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        model_name="grok-4",
        role="frontier_closed",
    ),
    ModelConfig(
        display_name="Kimi K2.5",
        provider="moonshot",
        endpoint="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        model_name="kimi-k2.5",
        role="frontier_supplementary",
    ),
    # Open-weight control row for the paper's Discussion ablation.
    # Together.ai exposes an OpenAI-compatible endpoint for Llama-3 family models.
    ModelConfig(
        display_name="Llama-3.3 70B (control)",
        provider="together",
        endpoint="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        role="open_weight_control",
    ),
]


def get_available_models() -> list[ModelConfig]:
    """Return only models whose API key is present in the environment.

    Logs a single ``Active: [...] | Skipped: [...]`` banner at INFO so the
    operator can confirm the run's effective scope at a glance. Per-key
    debug detail goes to DEBUG so re-calls don't spam the WARNING log.
    """
    available: list[ModelConfig] = []
    skipped: list[ModelConfig] = []
    for cfg in MODEL_REGISTRY:
        key = (os.getenv(cfg.api_key_env) or "").strip()
        if key:
            available.append(cfg)
        else:
            skipped.append(cfg)
            logger.debug("Skipping %s — %s not set", cfg.display_name, cfg.api_key_env)

    active_names = ", ".join(c.display_name for c in available) or "(none)"
    skipped_names = ", ".join(f"{c.display_name} [{c.api_key_env}]" for c in skipped) or "(none)"
    logger.info("Active models: %s | Skipped: %s", active_names, skipped_names)

    if not available:
        raise EnvironmentError(
            "No API keys found. Copy .env.example to .env and add at least one key."
        )
    return available


def build_target(config: ModelConfig) -> OpenAIChatTarget:
    """Construct a PyRIT OpenAIChatTarget from a model configuration.

    All registered providers expose OpenAI-compatible chat/completions
    endpoints (Anthropic via its OpenAI SDK compatibility layer, xAI and
    Together.ai natively). Optional ``temperature`` and ``extra_body`` from
    the ModelConfig are forwarded only when set.

    Raises:
        EnvironmentError: If the required API key is missing.
    """
    api_key = (os.getenv(config.api_key_env) or "").strip()
    if not api_key:
        raise EnvironmentError(
            f"{config.api_key_env} is not set. Add it to your .env file."
        )

    kwargs: dict[str, Any] = {
        "endpoint": config.endpoint,
        "api_key": api_key,
        "model_name": config.model_name,
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.extra_body is not None:
        kwargs["extra_body_parameters"] = config.extra_body

    return OpenAIChatTarget(**kwargs)
