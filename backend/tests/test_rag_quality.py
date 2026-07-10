from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.rag.reranker import rerank
from app.rag.retriever import HybridRetriever, KeywordRetriever, VectorRetriever


class _FakeEmbedder:
    async def embed_texts(self, texts: list[str], *, batch_size: int = 96) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@dataclass
class _FakeCollection:
    def query(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "ids": [["a", "b"]],
            "documents": [["totally unrelated content", "fastapi database connection"]],
            "metadatas": [[{"source": "doc1"}, {"source": "doc2"}]],
            "distances": [[0.01, 0.02]],
        }

    def get(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "ids": ["a", "b"],
            "documents": ["totally unrelated content", "fastapi database connection"],
            "metadatas": [{"source": "doc1"}, {"source": "doc2"}],
        }


class _FakeChromaManager:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(self, *, name: str):
        return self.collection


def test_rerank_improves_top1_relevance() -> None:
    mgr = _FakeChromaManager()
    vector = VectorRetriever(embedder=_FakeEmbedder(), chroma_manager=mgr, default_collection="devassist")
    keyword = KeywordRetriever(chroma_manager=mgr, default_collection="devassist")
    hybrid = HybridRetriever(vector=vector, keyword=keyword)

    query = "fastapi database"
    candidates = asyncio.run(hybrid.search(query=query, top_k=2))
    assert [c.id for c in candidates] == ["a", "b"]

    reranked = rerank(query=query, chunks=candidates, top_k=1)
    assert reranked[0].id == "b"

