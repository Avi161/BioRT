"""Anthropic OpenAI-compat chat target.

Anthropic exposes an OpenAI-compatible API at https://api.anthropic.com/v1/.
``OpenAIChatTarget`` works as-is, including ``response_format: {type: json_object}``
(natively supported by Anthropic's compat layer — no rewriting needed).

This subclass exists as a typed extension point so ``build_target`` can return
a distinct type for ``provider == "anthropic"``, enabling provider-specific
logic here in the future without touching the rest of the codebase.
"""

from __future__ import annotations

from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget


class AnthropicOpenAIChatTarget(OpenAIChatTarget):
    """``OpenAIChatTarget`` configured for ``https://api.anthropic.com/v1/``.

    Anthropic's OpenAI-compatible endpoint supports standard OpenAI request
    parameters, including ``response_format: {type: json_object}`` and
    ``response_format: {type: json_schema, ...}`` (with a subset of JSON Schema
    keywords — notably ``oneOf`` / ``anyOf`` / ``allOf`` are **not** supported).

    PyRIT's Crescendo and Scorer flows send ``response_format: {type: json_object}``
    to the adversarial LLM; Anthropic accepts this without modification, so no
    request-body rewriting is performed here.
    """
