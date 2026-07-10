from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.rag.reranker import rerank


@dataclass(frozen=True)
class _Chunk:
    id: str
    content: str
    metadata: dict[str, Any] | None = None


def test_rerank_orders_by_overlap() -> None:
    chunks = [
        _Chunk(id="1", content="this is unrelated"),
        _Chunk(id="2", content="fastapi database migration"),
        _Chunk(id="3", content="fastapi async database connection"),
    ]

    out = rerank(query="fastapi database", chunks=chunks, top_k=2)
    assert len(out) == 2
    assert out[0].id in {"2", "3"}
    assert out[0].score >= out[1].score
    assert all(x.score > 0 for x in out)

    out2 = rerank(query="fastapi database", chunks=chunks, top_k=2, min_score=999.0)
    assert out2 == []


def test_rerank_validates_inputs() -> None:
    with pytest.raises(ValueError):
        rerank(query=" ", chunks=[], top_k=5)
    with pytest.raises(ValueError):
        rerank(query="ok", chunks=[], top_k=0)
