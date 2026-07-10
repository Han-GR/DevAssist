from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.rag.bm25 import BM25Scorer
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


@dataclass(frozen=True)
class KeywordChunk:
    id: str
    content: str
    metadata: dict[str, Any] | None
    score: float


class KeywordRetriever:
    """
    关键词检索器（BM25，最小版本）。

    先把 collection 里的 documents 拉出来做 BM25 排序，适用于小规模知识库。
    """

    def __init__(self, *, chroma_manager: ChromaCollectionManager, default_collection: str) -> None:
        if not default_collection.strip():
            raise ValueError("default_collection is required")
        self._chroma_manager = chroma_manager
        self._default_collection = default_collection

    @classmethod
    def from_settings(cls, settings: Settings) -> "KeywordRetriever":
        chroma_manager = ChromaCollectionManager.from_settings(settings)
        return cls(chroma_manager=chroma_manager, default_collection=settings.chroma_collection)

    def search(
        self,
        *,
        query: str,
        top_k: int = 10,
        collection_name: str | None = None,
    ) -> list[KeywordChunk]:
        if not query.strip():
            raise ValueError("query is required")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        collection = collection_name or self._default_collection
        chroma_collection = self._chroma_manager.get_or_create_collection(name=collection)

        result = chroma_collection.get(include=["documents", "metadatas"])
        ids: list[str] = list(result.get("ids") or [])
        documents: list[str] = list(result.get("documents") or [])
        metadatas: list[dict[str, Any]] = list(result.get("metadatas") or [])

        if not ids or not documents:
            return []

        scorer = BM25Scorer(documents=[str(d) for d in documents])
        scores = scorer.score(query=query)
        scored = sorted(scores, key=lambda x: x.score, reverse=True)[:top_k]

        items: list[KeywordChunk] = []
        for s in scored:
            if s.score <= 0:
                continue
            i = s.doc_index
            meta = metadatas[i] if i < len(metadatas) else None
            items.append(
                KeywordChunk(
                    id=str(ids[i]),
                    content=str(documents[i]),
                    metadata=meta,
                    score=s.score,
                )
            )
        return items


@dataclass(frozen=True)
class HybridChunk:
    id: str
    content: str
    metadata: dict[str, Any] | None
    vector_distance: float | None
    bm25_score: float | None


class HybridRetriever:
    """
    混合检索（vector + BM25，最小版本）。
    """

    def __init__(self, *, vector: VectorRetriever, keyword: KeywordRetriever) -> None:
        self._vector = vector
        self._keyword = keyword

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridRetriever":
        return cls(vector=VectorRetriever.from_settings(settings), keyword=KeywordRetriever.from_settings(settings))

    async def search(
        self,
        *,
        query: str,
        top_k: int = 10,
        collection_name: str | None = None,
        vector_top_k: int | None = None,
        keyword_top_k: int | None = None,
    ) -> list[HybridChunk]:
        if not query.strip():
            raise ValueError("query is required")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        v_top = vector_top_k or top_k
        k_top = keyword_top_k or top_k

        vector_hits = await self._vector.search(query=query, top_k=v_top, collection_name=collection_name)
        keyword_hits = self._keyword.search(query=query, top_k=k_top, collection_name=collection_name)

        merged: dict[str, HybridChunk] = {}
        for h in vector_hits:
            merged[h.id] = HybridChunk(
                id=h.id,
                content=h.content,
                metadata=h.metadata,
                vector_distance=h.distance,
                bm25_score=None,
            )
        for h in keyword_hits:
            prev = merged.get(h.id)
            if prev is None:
                merged[h.id] = HybridChunk(
                    id=h.id,
                    content=h.content,
                    metadata=h.metadata,
                    vector_distance=None,
                    bm25_score=h.score,
                )
            else:
                merged[h.id] = HybridChunk(
                    id=prev.id,
                    content=prev.content,
                    metadata=prev.metadata,
                    vector_distance=prev.vector_distance,
                    bm25_score=h.score,
                )

        def _rank_key(item: HybridChunk) -> tuple[float, float]:
            d = item.vector_distance
            v = 0.0 if d is None else 1.0 / (1.0 + float(d))
            b = 0.0 if item.bm25_score is None else float(item.bm25_score)
            return (v, b)

        ranked = sorted(merged.values(), key=_rank_key, reverse=True)[:top_k]
        return ranked


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


async def hybrid_search(
    *, query: str, top_k: int = 10, collection_name: str | None = None
) -> list[HybridChunk]:
    """
    便捷入口：用默认 Settings 完成一次混合检索（vector + BM25）。

    Args:
        query (str): 用户查询。
        top_k (int): 返回数量，默认 10。
        collection_name (str | None): 指定 collection；不传则用 Settings 默认值。

    Returns:
        list[HybridChunk]: 混合检索结果列表。
    """
    settings = get_settings()
    retriever = HybridRetriever.from_settings(settings)
    return await retriever.search(query=query, top_k=top_k, collection_name=collection_name)
