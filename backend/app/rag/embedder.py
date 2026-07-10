from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI
import structlog

from app.core.config import Settings


class Embedder:
    """
    Embedding 客户端封装。

    目标：
    - 支持批量 embedding（减少网络往返）
    - 封装 OpenAI SDK 的细节，上层只关心输入文本和返回向量
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        """
        创建一个可复用的 embedding 客户端实例。

        Args:
            api_key (str): 调用 embedding 服务的 API Key（不能为空）。
            model (str): embedding 模型名，例如 "text-embedding-3-small"。
            base_url (str | None): OpenAI-style API 基地址；不传则使用 OpenAI 默认地址。
            client (Any | None): 允许注入自定义 client（主要用于测试）。

        Returns:
            None: 只初始化对象。

        Raises:
            ValueError: api_key 为空时抛出。

        Notes/Examples:
            Embedder 默认会复用底层 HTTP client，适合在应用启动时创建单例。
        """
        if not api_key:
            raise ValueError("Embedding api_key is required")

        self._model = model
        self._logger = structlog.get_logger()
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url or None)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Embedder":
        """
        从 Settings 构建 Embedder。

        Args:
            settings (Settings): 应用配置对象（包含 embedding_api_key / embedding_model / embedding_base_url）。

        Returns:
            Embedder: 根据 settings 构建好的客户端实例。

        Raises:
            ValueError: 当 embedding_api_key 为空时抛出。

        Notes/Examples:
            为了保持配置入口集中，业务层只依赖这个工厂方法即可。
        """
        return cls(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or None,
        )

    async def embed_texts(
        self, texts: list[str], *, batch_size: int = 64
    ) -> list[list[float]]:
        """
        对一组文本生成 embedding 向量（支持自动分批）。

        Args:
            texts (list[str]): 待 embedding 的文本列表。
            batch_size (int): 单次请求的批大小，默认 64。

        Returns:
            list[list[float]]: 与 texts 一一对应的向量列表。

        Raises:
            ValueError: batch_size 非正数时抛出。
            Exception: 底层网络/鉴权/供应商错误会原样抛出。

        Notes/Examples:
            - OpenAI-style embeddings API 支持 input 直接传 list，从而一次返回多条向量。
            - 返回顺序按 index 排好，确保稳定与 texts 对齐。
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not texts:
            return []

        start = time.perf_counter()
        results: list[list[float]] = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
                items = sorted(
                    getattr(response, "data", []),
                    key=lambda x: int(getattr(x, "index", 0)),
                )
                for item in items:
                    results.append(list(getattr(item, "embedding", [])))

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            usage = getattr(response, "usage", None) if "response" in locals() else None
            self._logger.info(
                "embedding_call",
                model=self._model,
                count=len(texts),
                latency_ms=elapsed_ms,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                success=True,
            )
            return results
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._logger.exception(
                "embedding_call",
                model=self._model,
                count=len(texts),
                latency_ms=elapsed_ms,
                success=False,
                error=str(exc),
            )
            raise
