from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Type, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Unified response dataclass returned by every provider
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Shared message helper
# ---------------------------------------------------------------------------


def dicts_to_lc_messages(messages: list[dict[str, str]]) -> list:
    """Convert [{role, content}] dicts to LangChain message objects."""
    lc = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            lc.append(AIMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    return lc


def extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a LangChain AIMessage (works for all providers)."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        return {
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "total_tokens": meta.get("total_tokens", 0),
        }
    return {}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """
    Common interface for all LLM providers.
    Every concrete provider must implement the four abstract methods.
    Agents interact exclusively with this interface — no provider SDK
    leaks into agent code.
    """

    # ------------------------------------------------------------------
    # Identity (required)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. 'claude', 'openai'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier, e.g. 'claude-sonnet-4-6'."""
        ...

    # ------------------------------------------------------------------
    # Core interface (required)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_chat_model(
        self,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Return a configured LangChain chat model.
        Used by agent tool-calling loops via bind_tools().
        """
        ...

    @abstractmethod
    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Single-turn invocation.
        Returns a unified LLMResponse regardless of underlying provider.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Yield response tokens one at a time."""
        ...

    @abstractmethod
    def structured_output(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str = (
            "You are a helpful AI assistant. "
            "Always respond with valid JSON that matches the requested schema."
        ),
    ) -> T:
        """
        Return a validated Pydantic model from a structured JSON response.
        Implementations may use JSON mode, tool-call schemas, or prompt engineering.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience (shared implementation)
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: list[Any],
        temperature: float = 0.0,
    ) -> BaseChatModel:
        """
        Return a chat model with LangChain tools bound.
        Used by all specialized agents for their ReAct tool-calling loops.
        """
        return self.get_chat_model(temperature=temperature).bind_tools(tools)

    def get_embeddings(self) -> Embeddings | None:
        """
        Return an embeddings instance, or None when the provider
        does not natively support text embeddings (e.g. Anthropic).
        """
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name} model={self.model_name}>"
