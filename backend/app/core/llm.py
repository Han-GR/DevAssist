from __future__ import annotations

import time
from typing import Any, Literal

from openai import AsyncOpenAI
import structlog

from app.core.config import Settings


class LLMClient:
    def __init__(
        self,
        *,
        provider: Literal["deepseek", "openai"],
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        # 这里强制校验一下：避免本地忘配 key，最后请求发出去才发现是 401/403
        if not api_key:
            raise ValueError("LLM api_key is required")

        self._provider = provider
        self._model = model
        self._logger = structlog.get_logger()

        # DeepSeek/OpenAI 都走 OpenAI-style 的接口，只需要切 base_url 就行
        resolved_base_url = base_url or self._default_base_url(provider)
        self._client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url)

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        return cls(
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
        )

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        stream: bool = False,
    ) -> Any:
        start = time.perf_counter()
        try:
            # 先按非流式做通
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            usage = getattr(response, "usage", None)
            # 这条日志主要为了后面做成本/性能分析：模型、耗时、token 用量
            self._logger.info(
                "llm_call",
                provider=self._provider,
                model=self._model,
                stream=stream,
                latency_ms=elapsed_ms,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None)
                if usage
                else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                success=True,
            )
            return response
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._logger.exception(
                "llm_call",
                provider=self._provider,
                model=self._model,
                stream=stream,
                latency_ms=elapsed_ms,
                success=False,
                error=str(exc),
            )
            raise

    @staticmethod
    def _default_base_url(provider: Literal["deepseek", "openai"]) -> str:
        if provider == "deepseek":
            return "https://api.deepseek.com/v1"
        return "https://api.openai.com/v1"
