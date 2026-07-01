from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Type, TypeVar

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
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

# Structured-output system suffix injected for every JSON request
_JSON_SUFFIX = (
    "\n\nIMPORTANT: Your response must be ONLY valid JSON with no explanation, "
    "no markdown fences, and no extra text."
)


class ClaudeProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.
    Default model: claude-sonnet-4-6 (configurable via CLAUDE_MODEL env var).
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model = model

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "claude"

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
        return ChatAnthropic(
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
        logger.debug(
            "claude.invoke",
            model=self._model,
            latency_ms=round(latency, 1),
            tokens=usage,
        )
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
        """
        Use Claude's with_structured_output via LangChain for type-safe JSON responses.
        Falls back to JSON prompt + Pydantic parse on failure.
        """
        llm = self.get_chat_model(temperature=0.0)

        try:
            structured_llm = llm.with_structured_output(schema)
            lc_messages = dicts_to_lc_messages([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ])
            result = structured_llm.invoke(lc_messages)
            if isinstance(result, schema):
                return result
        except Exception as exc:
            logger.debug("claude.structured_output.fallback", error=str(exc))

        # Fallback: plain JSON prompt
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        resp = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt + _JSON_SUFFIX,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return schema.model_validate(json.loads(clean))
