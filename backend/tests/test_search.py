from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.search as search_module


@dataclass(frozen=True)
class _HybridChunk:
    id: str
    content: str
    metadata: dict[str, Any] | None
    vector_distance: float | None
    bm25_score: float | None


@dataclass(frozen=True)
class _Reranked:
    id: str
    content: str
    metadata: dict[str, Any] | None
    score: float


def test_search_validates_query() -> None:
    client = TestClient(main_module.app)
    resp = client.post("/search", json={"query": "  "})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_query"


def test_search_returns_reranked_results(monkeypatch) -> None:
    async def _fake_hybrid_search(**kwargs):
        return [
            _HybridChunk(
                id="a",
                content="fastapi database",
                metadata={"source": "x"},
                vector_distance=0.01,
                bm25_score=1.2,
            ),
            _HybridChunk(
                id="b",
                content="unrelated",
                metadata={"source": "y"},
                vector_distance=0.02,
                bm25_score=0.0,
            ),
        ]

    def _fake_rerank(**kwargs):
        return [
            _Reranked(id="a", content="fastapi database", metadata={"source": "x"}, score=0.9),
        ]

    monkeypatch.setattr(search_module, "hybrid_search", _fake_hybrid_search)
    monkeypatch.setattr(search_module, "rerank", _fake_rerank)

    client = TestClient(main_module.app)
    resp = client.post("/search", json={"query": "fastapi", "top_k": 5})
    assert resp.status_code == 200

    body = resp.json()
    assert body["query"] == "fastapi"
    assert len(body["results"]) == 1
    assert body["results"][0]["id"] == "a"
    assert body["results"][0]["vector_distance"] == 0.01
    assert body["results"][0]["bm25_score"] == 1.2
