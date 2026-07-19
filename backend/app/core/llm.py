"""
LLM 客户端封装。

目标很简单：把不同厂商（DeepSeek/OpenAI）统一成“OpenAI-style”的调用方式，
上层业务只关心 messages、model、stream 等参数，不需要到处写适配逻辑。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from openai import AsyncOpenAI
import structlog

from app.core.config import Settings


class LLMClient:
    """
    统一的 LLM 调用入口。

    - DeepSeek / OpenAI 都使用 OpenAI SDK 的接口形状
    - 通过 base_url 切换不同供应商
    - 每次调用都会打结构化日志，方便后面做成本和性能分析
    """

    def __init__(
        self,
        *,
        provider: Literal["deepseek", "openai", "vllm"],
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        """
        创建一个可复用的 LLM 客户端实例。

        Args:
            provider (Literal["deepseek", "openai"]): 模型供应商类型，用于选择默认 base_url 和打点字段。
            api_key (str): 调用 LLM 的 API Key（不能为空）。
            model (str): 模型名，例如 "deepseek-chat"。
            base_url (str | None): OpenAI-style API 基地址；不传则按 provider 取默认值。

        Returns:
            None: 只初始化对象，不返回业务数据。

        Raises:
            ValueError: api_key 为空时抛出，避免请求发出去才发现鉴权失败。

        Notes:
            这个类偏向“长期复用”而不是“每次请求都 new 一个”，主要是为了避免频繁创建底层 HTTP client 的开销。
        """
        # 这里强制校验一下：避免本地忘配 key，最后请求发出去才发现是 401/403
        if not api_key:
            raise ValueError("LLM api_key is required")

        self._provider = provider
        self._api_key = api_key
        self._model = model
        self._logger = structlog.get_logger()

        # DeepSeek/OpenAI 都走 OpenAI-style 的接口，只需要切 base_url 就行
        resolved_base_url = base_url or self._default_base_url(provider)
        self._base_url = resolved_base_url
        self._client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url)

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        """
        从 Settings 构建 LLMClient。

        Args:
            settings (Settings): 应用配置对象（包含 provider/api_key/model/base_url）。

        Returns:
            LLMClient: 根据 settings 构建好的客户端实例。

        Raises:
            ValueError: 当 settings 中 api_key 等关键字段为空时，底层初始化会抛出。

        Notes:
            入口集中在这里，后续如果需要扩展配置（比如 timeout、proxy），可以在这一处完成，不影响业务层调用。
        """
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
        model: str | None = None,
        stream: bool = False,
    ) -> Any:
        """
        发起一次聊天补全请求。

        Args:
            messages (list[dict[str, Any]]): OpenAI-style messages 列表（role/content）。
            temperature (float): 采样温度，越高越“发散”。
            model (str | None): 可选，覆盖默认模型名（用于评测或临时切换）。
            stream (bool): 是否启用流式；为 True 时返回可迭代的 chunk 流。

        Returns:
            Any: OpenAI SDK 的原始 response。stream=False 返回完整响应；stream=True 返回可 async for 的流对象。

        Raises:
            Exception: 底层网络/鉴权/供应商错误都会原样抛出；同时会记录 llm_call 日志（success=False）。

        Notes:
            返回值保持 OpenAI SDK 的原始结构，主要是为了上层代码不用做二次适配：
            - 非流式直接读 response.choices[0].message.content
            - 流式直接 async for chunk，取 chunk.choices[0].delta.content
        """
        start = time.perf_counter()
        used_model = str(model) if model else self._model
        try:
            # 先按非流式做通
            response = await self._client.chat.completions.create(
                model=used_model,
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
                model=used_model,
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
                model=used_model,
                stream=stream,
                latency_ms=elapsed_ms,
                success=False,
                error=str(exc),
            )
            raise

    @staticmethod
    def _default_base_url(provider: Literal["deepseek", "openai", "vllm"]) -> str:
        """
        不显式传 base_url 时的兜底地址。

        Args:
            provider (Literal["deepseek", "openai"]): 供应商类型。

        Returns:
            str: 对应供应商的默认 OpenAI-style base_url。

        Raises:
            None

        Notes:
            这个函数只做最基础的默认值选择，避免把厂商地址散落在代码各处。
        """
        if provider == "deepseek":
            return "https://api.deepseek.com/v1"
        if provider == "vllm":
            return "http://localhost:8002/v1"
        return "https://api.openai.com/v1"

    def with_model(self, *, model: str) -> "LLMClient":
        """
        基于当前客户端创建一个“只覆盖模型名”的新实例。

        Args:
            model (str): 新的默认模型名（例如 vLLM 的 LoRA adapter id）。

        Returns:
            LLMClient: 新的客户端实例（复用同一 provider/base_url/api_key，但默认模型不同）。
        """

        return LLMClient(
            provider=self._provider,
            api_key=self._api_key,
            model=str(model),
            base_url=self._base_url,
        )
