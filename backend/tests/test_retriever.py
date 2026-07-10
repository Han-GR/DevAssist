from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from app.rag.retriever import RetrievedChunk, VectorRetriever


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str], *, batch_size: int = 96) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


@dataclass
class _FakeCollection:
    last_query: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_query = kwargs
        return {
            "ids": [["a", "b"]],
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [[{"k": "v1"}, {"k": "v2"}]],
            "distances": [[0.01, 0.02]],
        }


class _FakeChromaManager:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(self, *, name: str):
        return self.collection


def test_vector_retriever_search_returns_chunks() -> None:
    embedder = _FakeEmbedder()
    mgr = _FakeChromaManager()
    r = VectorRetriever(embedder=embedder, chroma_manager=mgr, default_collection="devassist")

    chunks = asyncio.run(r.search(query="hello", top_k=2))
    assert chunks == [
        RetrievedChunk(id="a", content="doc-a", metadata={"k": "v1"}, distance=0.01),
        RetrievedChunk(id="b", content="doc-b", metadata={"k": "v2"}, distance=0.02),
    ]

    assert embedder.calls == [["hello"]]
    assert mgr.collection.last_query
    assert mgr.collection.last_query["n_results"] == 2


def test_vector_retriever_validates_inputs() -> None:
    r = VectorRetriever(
        embedder=_FakeEmbedder(),
        chroma_manager=_FakeChromaManager(),
        default_collection="devassist",
    )
    with pytest.raises(ValueError):
        asyncio.run(r.search(query="  "))
    with pytest.raises(ValueError):
        asyncio.run(r.search(query="hi", top_k=0))
