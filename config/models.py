"""Model registry and PyRIT target factory.

Centralizes model configurations so hello_world.py and matrix_runner.py
share the same definitions. Each model maps to an OpenAI-compatible
endpoint; Anthropic uses a LiteLLM proxy wrapper (deferred until key is added).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pyrit.prompt_target import OpenAIChatTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    provider: str
    endpoint: str
    api_key_env: str
    model_name: str

MODEL_REGISTRY: list[ModelConfig] = [
    ModelConfig(
        display_name="Gemini 3 Pro",
        provider="google",
        endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        model_name="gemini-3-pro-preview",
    ),
    ModelConfig(
        display_name="DeepSeek V4 Flash",
        provider="deepseek",
        endpoint="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model_name="deepseek-chat",
    ),
    ModelConfig(
        display_name="Kimi 2.5",
        provider="moonshot",
        endpoint="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        model_name="kimi-2.5",
    ),
    ModelConfig(
        display_name="GPT-5.4",
        provider="openai",
        endpoint="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_name="gpt-5.4",
    ),
    ModelConfig(
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        endpoint="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        model_name="claude-sonnet-4-6-20250514",
    ),
]


def get_available_models() -> list[ModelConfig]:
    """Return only models whose API key is present in the environment."""
    available: list[ModelConfig] = []
    for cfg in MODEL_REGISTRY:
        key = os.getenv(cfg.api_key_env)
        if key:
            available.append(cfg)
        else:
            logger.warning("Skipping %s — %s not set", cfg.display_name, cfg.api_key_env)
    if not available:
        raise EnvironmentError(
            "No API keys found. Copy .env.example to .env and add at least one key."
        )
    return available


def build_target(config: ModelConfig) -> OpenAIChatTarget:
    """Construct a PyRIT OpenAIChatTarget from a model configuration.

    All supported providers expose OpenAI-compatible chat/completions
    endpoints, so a single target class works for all of them.

    Args:
        config: A ModelConfig with endpoint, key env var, and model name.

    Returns:
        A configured OpenAIChatTarget ready for orchestrators.

    Raises:
        EnvironmentError: If the required API key is missing.
    """
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"{config.api_key_env} is not set. Add it to your .env file."
        )

    # Anthropic's native API is not OpenAI-compatible.
    # TODO(phase2): wrap with LiteLLM proxy or a custom PromptChatTarget.
    if config.provider == "anthropic":
        raise NotImplementedError(
            f"{config.display_name} requires a LiteLLM proxy wrapper. "
            "This will be implemented when the Anthropic key is added."
        )

    return OpenAIChatTarget(
        endpoint=config.endpoint,
        api_key=api_key,
        model_name=config.model_name,
    )
