from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Type, TypeVar

import structlog
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from openai import AzureOpenAI
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


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI provider.
    Supports GPT-4o and other models deployed on Azure OpenAI Service.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-08-01-preview",
        embedding_deployment: str = "text-embedding-3-large",
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._deployment = deployment
        self._api_version = api_version
        self._embedding_deployment = embedding_deployment

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def model_name(self) -> str:
        return self._deployment

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
        return AzureChatOpenAI(
            azure_deployment=self._deployment,
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
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
        logger.debug("azure.invoke", deployment=self._deployment, latency_ms=round(latency, 1), tokens=usage)
        return LLMResponse(
            content=str(response.content),
            provider=self.provider_name,
            model=self._deployment,
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
        """Use Azure OpenAI JSON mode for reliable structured responses."""
        client = AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )
        response = client.chat.completions.create(
            model=self._deployment,
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
        return AzureOpenAIEmbeddings(
            azure_deployment=self._embedding_deployment,
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )
