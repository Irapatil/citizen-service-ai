from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Type, TypeVar

import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.services.llm.base_provider import (
    BaseLLMProvider,
    LLMResponse,
    dicts_to_lc_messages,
    extract_usage,
)

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider.
    Default model: gpt-4o (configurable via OPENAI_MODEL env var).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Chat model
    # ------------------------------------------------------------------

    def get_chat_model(
        self,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        return ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Invoke
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        llm = self.get_chat_model(temperature=temperature, max_tokens=max_tokens)
        lc_messages = dicts_to_lc_messages(messages)

        t0 = time.monotonic()
        response = await llm.ainvoke(lc_messages)
        latency = (time.monotonic() - t0) * 1000

        usage = extract_usage(response)
        logger.debug("openai.invoke", model=self._model, latency_ms=round(latency, 1), tokens=usage)
        return LLMResponse(
            content=str(response.content),
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        llm = self.get_chat_model(temperature=temperature, max_tokens=max_tokens, streaming=True)
        lc_messages = dicts_to_lc_messages(messages)
        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                yield str(chunk.content)

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def structured_output(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str = (
            "You are a helpful AI assistant. "
            "Always respond with valid JSON that matches the requested schema."
        ),
    ) -> T:
        """Use OpenAI JSON mode for reliable structured responses."""
        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(raw))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(
            model=self._embedding_model,
            api_key=self._api_key,
        )
