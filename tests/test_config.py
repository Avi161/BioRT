"""Tests for config/models.py — model registry and target factory."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from config.models import (
    MODEL_REGISTRY,
    ModelConfig,
    build_target,
    get_available_models,
)


class TestModelRegistry:
    def test_registry_has_at_least_five_models(self) -> None:
        assert len(MODEL_REGISTRY) >= 5

    def test_all_entries_are_model_configs(self) -> None:
        for cfg in MODEL_REGISTRY:
            assert isinstance(cfg, ModelConfig)

    def test_display_names_are_unique(self) -> None:
        names = [cfg.display_name for cfg in MODEL_REGISTRY]
        assert len(names) == len(set(names))

    def test_all_endpoints_are_https(self) -> None:
        for cfg in MODEL_REGISTRY:
            assert cfg.endpoint.startswith("https://"), (
                f"{cfg.display_name} endpoint is not HTTPS"
            )

    def test_all_api_key_envs_end_with_key(self) -> None:
        for cfg in MODEL_REGISTRY:
            assert cfg.api_key_env.endswith("_KEY") or cfg.api_key_env.endswith("_API_KEY")

    def test_includes_open_weight_control(self) -> None:
        roles = {cfg.role for cfg in MODEL_REGISTRY}
        assert "open_weight_control" in roles, (
            "Registry must include an open-weight control row for the paper's ablation"
        )


class TestGetAvailableModels:
    def test_returns_models_with_keys(self) -> None:
        env = {"DEEPSEEK_API_KEY": "sk-test", "GEMINI_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=False):
            available = get_available_models()
            names = {cfg.display_name for cfg in available}
            assert "DeepSeek V4 Flash" in names
            assert "Gemini 3 Pro" in names

    def test_raises_when_no_keys(self) -> None:
        empty_env = {cfg.api_key_env: "" for cfg in MODEL_REGISTRY}
        with patch.dict(os.environ, empty_env, clear=True):
            with pytest.raises(EnvironmentError, match="No API keys found"):
                get_available_models()


class TestBuildTarget:
    def test_missing_key_raises_environment_error(self) -> None:
        deepseek_cfg = next(c for c in MODEL_REGISTRY if c.provider == "deepseek")
        with patch.dict(os.environ, {deepseek_cfg.api_key_env: ""}, clear=True):
            with pytest.raises(EnvironmentError, match="not set"):
                build_target(deepseek_cfg)

    def test_deepseek_temperature_zero(self) -> None:
        """DeepSeek must default to temperature=0 to suppress thinking-mode tokens
        when used as the adversary/scorer LLM."""
        deepseek_cfg = next(c for c in MODEL_REGISTRY if c.provider == "deepseek")
        assert deepseek_cfg.temperature == 0.0
