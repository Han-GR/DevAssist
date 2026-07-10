from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import Settings


class ChromaCollectionManager:
    """
    Chroma collection 管理器。

    目标：
    - 把 Chroma 的连接和 collection 获取/创建统一放在一个入口
    - 让后续 ingestion / retrieval 代码只关心“往哪个 collection 写/查”
    """

    def __init__(self, *, host: str, port: int, client: Any | None = None) -> None:
        """
        创建一个 collection 管理器。

        Args:
            host (str): Chroma 服务地址（docker compose 下通常是 "chroma"）。
            port (int): Chroma 服务端口（默认 8000）。
            client (Any | None): 可注入 client，方便测试时不走真实网络。

        Returns:
            None: 只初始化对象。

        Raises:
            ValueError: port 非法时抛出。

        Notes/Examples:
            - Chroma 的 Python client 是同步接口；我们先以“业务层同步调用”为主，
              后续如果需要强 async，可以放到线程池里跑。
        """
        if port <= 0:
            raise ValueError("port must be a positive integer")

        self._client = client or chromadb.HttpClient(host=host, port=port)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChromaCollectionManager":
        """
        从 Settings 构建 ChromaCollectionManager。

        Args:
            settings (Settings): 应用配置对象（包含 chroma_host / chroma_port）。

        Returns:
            ChromaCollectionManager: 管理器实例。

        Raises:
            ValueError: chroma_port 非法时抛出。

        Notes/Examples:
            为了保持配置入口集中，业务层只依赖这个工厂方法即可。
        """
        return cls(host=settings.chroma_host, port=settings.chroma_port)

    def get_or_create_collection(self, *, name: str) -> Collection:
        """
        获取或创建 collection。

        Args:
            name (str): collection 名称。

        Returns:
            Collection: 可用的 collection 对象。

        Raises:
            ValueError: name 为空时抛出。
            Exception: 连接异常或 Chroma 服务端错误会原样抛出。

        Notes/Examples:
            Chroma 会在 collection 不存在时自动创建；这里显式调用是为了行为更清晰。
        """
        if not name.strip():
            raise ValueError("collection name is required")

        return self._client.get_or_create_collection(name=name)
