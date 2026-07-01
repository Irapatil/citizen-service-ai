from __future__ import annotations

import structlog
from functools import lru_cache

from backend.config import get_settings
from backend.services.llm.base_provider import BaseLLMProvider

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_provider() -> BaseLLMProvider:
    """
    Factory function — reads LLM_PROVIDER from settings and returns
    the matching provider singleton.

    Supported values:
        claude   → Anthropic Claude (default)
        openai   → OpenAI GPT-4o
        azure    → Azure OpenAI
        gemini   → Google Gemini

    Usage in any agent:
        provider = get_llm_provider()
        response = await provider.invoke(messages)
        llm_with_tools = provider.bind_tools(tools)
    """
    settings = get_settings()
    provider_name = settings.llm_provider.lower().strip()

    logger.info("llm_provider.init", provider=provider_name)

    if provider_name == "claude":
        from backend.services.llm.claude_provider import ClaudeProvider
        if not settings.anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file."
            )
        return ClaudeProvider(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )

    if provider_name == "openai":
        from backend.services.llm.openai_provider import OpenAIProvider
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    if provider_name == "azure":
        from backend.services.llm.azure_openai_provider import AzureOpenAIProvider
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise ValueError(
                "LLM_PROVIDER=azure but AZURE_OPENAI_API_KEY or "
                "AZURE_OPENAI_ENDPOINT is not set."
            )
        return AzureOpenAIProvider(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_deployment_name,
            api_version=settings.azure_openai_api_version,
            embedding_deployment=settings.azure_openai_embedding_deployment,
        )

    if provider_name == "gemini":
        from backend.services.llm.gemini_provider import GeminiProvider
        if not settings.google_api_key:
            raise ValueError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set."
            )
        return GeminiProvider(
            api_key=settings.google_api_key,
            model=settings.gemini_model,
        )

    if provider_name == "mock":
        from backend.services.llm.mock_provider import MockLLMProvider
        logger.info("llm_provider.mock_mode")
        return MockLLMProvider()

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider_name}'. "
        "Valid options: claude, openai, azure, gemini, mock"
    )


def get_embedding_provider():
    """
    Return an embeddings instance for ChromaDB.

    Priority:
      1. The active LLM provider's embeddings (if it supports them).
      2. OpenAI embeddings as universal fallback (e.g. when using Claude).

    The fallback requires OPENAI_API_KEY to be set.
    """
    provider = get_llm_provider()
    embeddings = provider.get_embeddings()
    if embeddings is not None:
        return embeddings

    # Fallback: OpenAI embeddings (Anthropic has no embedding API)
    settings = get_settings()
    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        logger.info("embedding.fallback_openai")
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=settings.openai_api_key,
        )

    # Last resort: Azure embeddings
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        from langchain_openai import AzureOpenAIEmbeddings
        logger.info("embedding.fallback_azure")
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_openai_embedding_deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    logger.warning(
        "embedding.no_provider",
        msg="No embedding provider configured — ChromaDB will be disabled.",
    )
    return None
