"""
Backward-compatibility shim.

All new code should import from backend.services.llm instead:

    from backend.services.llm import get_llm_provider

    provider = get_llm_provider()
    response = await provider.invoke(messages)
    llm_with_tools = provider.bind_tools(tools)

This module is kept so any external scripts that imported the old
helper functions continue to work without modification.
"""
from __future__ import annotations

from typing import Any

from backend.services.llm import get_llm_provider, get_embedding_provider


def get_llm(**kwargs):
    """Return the active provider's chat model. Deprecated — use get_llm_provider()."""
    return get_llm_provider().get_chat_model(**kwargs)


def get_embeddings():
    """Return the active embedding model. Deprecated — use get_embedding_provider()."""
    return get_embedding_provider()


async def invoke_llm(llm: Any, messages: list[dict]) -> tuple[str, dict]:
    """Deprecated wrapper. Use provider.invoke() directly."""
    response = await get_llm_provider().invoke(messages)
    return response.content, response.usage


def tool_calling(llm: Any, tools: list[Any]) -> Any:
    """Deprecated wrapper. Use provider.bind_tools() directly."""
    return get_llm_provider().bind_tools(tools)


def structured_output(prompt: str, response_model: Any, system_prompt: str = "") -> Any:
    """Deprecated wrapper. Use provider.structured_output() directly."""
    return get_llm_provider().structured_output(prompt, response_model, system_prompt)
