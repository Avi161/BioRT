"""Tests for Anthropic OpenAI-compat JSON response_format handling."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from pyrit.models.json_response_config import _JsonResponseConfig

from config.anthropic_openai_chat_target import AnthropicOpenAIChatTarget


def test_json_object_passes_through_unchanged() -> None:
    """Anthropic's compat layer supports json_object natively; no conversion needed."""

    async def _exercise() -> None:
        t = object.__new__(AnthropicOpenAIChatTarget)
        mock_body = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        }
        with patch(
            "pyrit.prompt_target.openai.openai_chat_target.OpenAIChatTarget._construct_request_body",
            new=AsyncMock(return_value=dict(mock_body)),
        ):
            out = await AnthropicOpenAIChatTarget._construct_request_body(
                t,
                conversation=[],
                json_config=_JsonResponseConfig(),
            )
        assert out["response_format"] == {"type": "json_object"}

    asyncio.run(_exercise())


def test_passthrough_when_parent_already_uses_json_schema() -> None:
    """Do not clobber a proper json_schema from PyRIT (e.g. schema in metadata)."""

    async def _exercise() -> None:
        t = object.__new__(AnthropicOpenAIChatTarget)
        original: dict = {
            "type": "json_schema",
            "json_schema": {
                "name": "custom",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                "strict": True,
            },
        }
        mock_body = {"model": "m", "messages": [], "response_format": original}
        with patch(
            "pyrit.prompt_target.openai.openai_chat_target.OpenAIChatTarget._construct_request_body",
            new=AsyncMock(return_value=dict(mock_body)),
        ):
            out = await AnthropicOpenAIChatTarget._construct_request_body(
                t, conversation=[], json_config=_JsonResponseConfig()
            )
        assert out["response_format"] == original

    asyncio.run(_exercise())


def test_anthropic_target_is_subclass_of_openai_chat_target() -> None:
    """AnthropicOpenAIChatTarget must inherit from OpenAIChatTarget for PyRIT compat."""
    from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget

    assert issubclass(AnthropicOpenAIChatTarget, OpenAIChatTarget)


@pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY", "").strip() and os.getenv("ANTHROPIC_SMOKE", "")),
    reason="Set ANTHROPIC_SMOKE=1 and ANTHROPIC_API_KEY to run one live HTTP check",
)
def test_live_anthropic_accepts_json_object_smoke() -> None:
    """Minimal chat.completions with json_object response_format (costs a small API call)."""
    from openai import AsyncOpenAI

    from config.models import MODEL_REGISTRY
    from dotenv import load_dotenv

    async def _call() -> None:
        load_dotenv()
        anth = next(c for c in MODEL_REGISTRY if c.provider == "anthropic")
        key = (os.getenv(anth.api_key_env) or "").strip()
        assert key, "ANTHROPIC_API_KEY should be set for this test"
        client = AsyncOpenAI(api_key=key, base_url=anth.endpoint.rstrip("/") + "/")
        resp = await client.chat.completions.create(
            model=anth.model_name,
            max_tokens=32,
            messages=[{"role": "user", "content": 'Reply with JSON: {"ok": true}'}],
            response_format={"type": "json_object"},  # type: ignore[arg-type]
        )
        assert resp.choices[0].message

    asyncio.run(_call())
