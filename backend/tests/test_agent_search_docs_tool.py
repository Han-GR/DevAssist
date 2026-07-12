from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.builtin_tools import SEARCH_DOCS_SNIPPET_CHARS, create_search_docs_tool
from app.core.errors import AppError
from app.rag.retriever import HybridChunk


def test_search_docs_tool_invalid_top_k_raises_app_error() -> None:
    tool = create_search_docs_tool()
    with pytest.raises(AppError) as exc:
        asyncio.run(tool.call({"query": "x", "top_k": 0}))
    assert exc.value.code == "tool_input_invalid"


def test_search_docs_tool_calls_hybrid_search_and_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_hybrid_search(*, query: str, top_k: int, collection_name: str | None = None) -> list[HybridChunk]:
        assert query == "fastapi"
        assert top_k == 2
        assert collection_name == "fastapi_docs"
        return [
            HybridChunk(
                id="1",
                content="hello",
                metadata={"source": "a.md", "chunk_index": 3},
                vector_distance=0.1,
                bm25_score=None,
            ),
            HybridChunk(
                id="2",
                content="x" * (SEARCH_DOCS_SNIPPET_CHARS + 10),
                metadata={"source": "b.md", "chunk_index": 4},
                vector_distance=None,
                bm25_score=2.5,
            ),
        ]

    import app.agent.builtin_tools as module

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)

    tool = create_search_docs_tool()
    out = asyncio.run(
        tool.call({"query": "fastapi", "top_k": 2, "collection_name": "fastapi_docs"})
    )

    assert "results" in out
    assert len(out["results"]) == 2

    first = out["results"][0]
    assert first["id"] == "1"
    assert first["source"] == "a.md"
    assert first["chunk_index"] == 3
    assert first["content"] == "hello"
    assert first["vector_distance"] == 0.1
    assert first["bm25_score"] == 0.0

    second = out["results"][1]
    assert second["id"] == "2"
    assert len(second["content"]) == SEARCH_DOCS_SNIPPET_CHARS
    assert second["vector_distance"] == 0.0
    assert second["bm25_score"] == 2.5

