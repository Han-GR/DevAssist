from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.rag.chroma import ChromaCollectionManager
from app.rag.embedder import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    """
    检索返回的 chunk。

    这里不做复杂抽象，只保留后续生成需要的最小字段。
    """

    id: str
    content: str
    metadata: dict[str, Any] | None
    distance: float | None


class VectorRetriever:
    """
    向量相似度检索器（最小版本）。

    工作流：
    - 对 query 做 embedding
    - 在 Chroma collection 里按向量相似度检索 Top-K
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        chroma_manager: ChromaCollectionManager,
        default_collection: str,
    ) -> None:
        """
        创建向量检索器。

        Args:
            embedder (Embedder): embedding 客户端。
            chroma_manager (ChromaCollectionManager): Chroma collection 管理器。
            default_collection (str): 默认检索的 collection 名称。

        Raises:
            ValueError: default_collection 为空时抛出。
        """
        if not default_collection.strip():
            raise ValueError("default_collection is required")

        self._embedder = embedder
        self._chroma_manager = chroma_manager
        self._default_collection = default_collection

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorRetriever":
        """
        从 Settings 构建 VectorRetriever。

        Args:
            settings (Settings): 应用配置对象。

        Returns:
            VectorRetriever: 可直接用于检索的实例。
        """
        embedder = Embedder.from_settings(settings)
        chroma_manager = ChromaCollectionManager.from_settings(settings)
        return cls(
            embedder=embedder,
            chroma_manager=chroma_manager,
            default_collection=settings.chroma_collection,
        )

    async def search(
        self,
        *,
        query: str,
        top_k: int = 10,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        按向量相似度检索相关 chunk。

        Args:
            query (str): 用户查询。
            top_k (int): 返回数量，默认 10。
            collection_name (str | None): 指定 collection；不传则用默认值。

        Returns:
            list[RetrievedChunk]: 检索结果列表，按相似度从高到低排列（Chroma 的顺序）。

        Raises:
            ValueError: query 为空或 top_k 非正数时抛出。
            Exception: embedding/chroma 调用失败时原样抛出。
        """
        if not query.strip():
            raise ValueError("query is required")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        collection = collection_name or self._default_collection
        chroma_collection = self._chroma_manager.get_or_create_collection(name=collection)

        vectors = await self._embedder.embed_texts([query], batch_size=1)
        query_vector = vectors[0]

        result = chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        items: list[RetrievedChunk] = []
        for i in range(min(len(ids), len(documents))):
            meta = metadatas[i] if i < len(metadatas) else None
            dist = distances[i] if i < len(distances) else None
            items.append(
                RetrievedChunk(
                    id=str(ids[i]),
                    content=str(documents[i]),
                    metadata=meta,
                    distance=dist,
                )
            )

        return items


async def search(*, query: str, top_k: int = 10, collection_name: str | None = None) -> list[RetrievedChunk]:
    """
    便捷入口：用默认 Settings 完成一次向量检索。

    Args:
        query (str): 用户查询。
        top_k (int): 返回数量，默认 10。
        collection_name (str | None): 指定 collection；不传则用 Settings 默认值。

    Returns:
        list[RetrievedChunk]: 检索结果列表。
    """
    settings = get_settings()
    retriever = VectorRetriever.from_settings(settings)
    return await retriever.search(query=query, top_k=top_k, collection_name=collection_name)

