from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.agent.memory import LongTermMemory, MemoryManager, ShortTermMemory


class _FakeEmbedder:
    async def embed_texts(self, texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
        _ = batch_size
        return [[float(len(t))] for t in texts]


@dataclass
class _StoredDoc:
    doc_id: str
    document: str
    embedding: list[float]
    metadata: dict[str, Any]


class _FakeCollection:
    def __init__(self, *, store: list[_StoredDoc]) -> None:
        self._store = store

    def add(self, *, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
        for doc_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas, strict=True):
            self._store.append(_StoredDoc(doc_id=doc_id, document=doc, embedding=emb, metadata=dict(meta)))

    def query(self, *, query_embeddings: list[list[float]], n_results: int, where: dict[str, Any]) -> dict[str, Any]:
        _ = query_embeddings
        conv_id = str(where.get("conversation_id", ""))
        docs = [d.document for d in self._store if str(d.metadata.get("conversation_id")) == conv_id]
        return {"documents": [docs[:n_results]]}


class _FakeChromaManager:
    def __init__(self, *, collection: _FakeCollection) -> None:
        self._collection = collection

    def get_or_create_collection(self, *, name: str) -> _FakeCollection:
        _ = name
        return self._collection


def test_long_term_memory_survives_new_manager_instance() -> None:
    shared_store: list[_StoredDoc] = []
    collection = _FakeCollection(store=shared_store)
    chroma = _FakeChromaManager(collection=collection)
    embedder = _FakeEmbedder()

    cid = uuid4()

    ltm1 = LongTermMemory(embedder=embedder, chroma=chroma, collection_name="agent_memory")  # type: ignore[arg-type]
    asyncio.run(ltm1.add_summary(conversation_id=cid, summary="S1"))

    ltm2 = LongTermMemory(embedder=embedder, chroma=chroma, collection_name="agent_memory")  # type: ignore[arg-type]
    short2 = ShortTermMemory(max_turns=1, max_conversations=10)
    mgr2 = MemoryManager(short_term=short2, long_term=ltm2, summarize_min_messages=2)

    history = asyncio.run(mgr2.build_history(conversation_id=cid, query="q"))
    assert history
    assert history[0]["role"] == "system"
    assert "- S1" in history[0]["content"]


def test_long_term_memory_is_isolated_by_conversation_id() -> None:
    shared_store: list[_StoredDoc] = []
    collection = _FakeCollection(store=shared_store)
    chroma = _FakeChromaManager(collection=collection)
    embedder = _FakeEmbedder()

    cid1 = uuid4()
    cid2 = uuid4()

    ltm = LongTermMemory(embedder=embedder, chroma=chroma, collection_name="agent_memory")  # type: ignore[arg-type]
    asyncio.run(ltm.add_summary(conversation_id=cid1, summary="ONLY_CID1"))

    short = ShortTermMemory(max_turns=1, max_conversations=10)
    mgr = MemoryManager(short_term=short, long_term=ltm, summarize_min_messages=2)

    history1 = asyncio.run(mgr.build_history(conversation_id=cid1, query="q"))
    history2 = asyncio.run(mgr.build_history(conversation_id=cid2, query="q"))

    assert history1 and "- ONLY_CID1" in history1[0]["content"]
    assert history2 == []

