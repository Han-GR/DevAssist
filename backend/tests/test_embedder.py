from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.embedder import Embedder


@dataclass
class _FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class _FakeUsage:
    prompt_tokens: int
    total_tokens: int


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]
    usage: _FakeUsage | None = None


class _FakeEmbeddingsApi:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]) -> _FakeEmbeddingResponse:
        self.calls.append(input)
        data: list[_FakeEmbeddingItem] = []
        for i, text in enumerate(input):
            data.append(_FakeEmbeddingItem(index=i, embedding=[float(len(text)), float(i)]))
        return _FakeEmbeddingResponse(data=data, usage=_FakeUsage(prompt_tokens=1, total_tokens=1))


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsApi()


def test_embed_texts_batches_and_keeps_order() -> None:
    fake = _FakeClient()
    embedder = Embedder(api_key="x", model="m", client=fake)

    texts = ["a", "bb", "ccc"]
    vectors = asyncio.run(embedder.embed_texts(texts, batch_size=2))

    assert fake.embeddings.calls == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]


def test_embed_texts_empty_returns_empty() -> None:
    fake = _FakeClient()
    embedder = Embedder(api_key="x", model="m", client=fake)
    assert asyncio.run(embedder.embed_texts([])) == []


def test_embed_texts_invalid_batch_size_raises() -> None:
    fake = _FakeClient()
    embedder = Embedder(api_key="x", model="m", client=fake)
    with pytest.raises(ValueError):
        asyncio.run(embedder.embed_texts(["a"], batch_size=0))
