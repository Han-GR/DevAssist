from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.chroma import ChromaCollectionManager
from app.rag.embedder import Embedder
from app.rag.splitter import split_text_semantic


@dataclass
class _FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddingsApi:
    async def create(self, *, model: str, input: list[str]):
        data: list[_FakeEmbeddingItem] = []
        for i, text in enumerate(input):
            data.append(_FakeEmbeddingItem(index=i, embedding=[float(len(text)), float(i)]))
        return _FakeEmbeddingResponse(data=data)


class _FakeOpenAiClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsApi()


class _FakeCollection:
    def __init__(self, *, name: str) -> None:
        self.name = name
        self.last_add: dict[str, object] | None = None

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        self.last_add = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, *, name: str):
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name=name)
        return self.collections[name]


def test_rag_minimal_pipeline_split_embed_store() -> None:
    text = (
        "第一段。第二句。\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "结尾段落。"
    )
    chunks = split_text_semantic(text, chunk_size=60, overlap=0)
    assert chunks

    embedder = Embedder(api_key="x", model="embedding-3", client=_FakeOpenAiClient())
    vectors = asyncio.run(embedder.embed_texts(chunks, batch_size=2))
    assert len(vectors) == len(chunks)

    chroma_client = _FakeChromaClient()
    mgr = ChromaCollectionManager(host="x", port=8000, client=chroma_client)
    col = mgr.get_or_create_collection(name="devassist")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "unit-test", "index": i} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    assert col.last_add
    assert col.last_add["ids"] == ids
    assert col.last_add["documents"] == chunks
    assert len(col.last_add["embeddings"]) == len(chunks)
