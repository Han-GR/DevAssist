from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.rag.generator as generator_module


@dataclass(frozen=True)
class _HybridChunk:
    id: str
    content: str
    metadata: dict[str, Any] | None
    vector_distance: float | None = None
    bm25_score: float | None = None


@dataclass(frozen=True)
class _Reranked:
    id: str
    content: str
    metadata: dict[str, Any] | None
    score: float


@dataclass(frozen=True)
class _Msg:
    content: str


@dataclass(frozen=True)
class _Choice:
    message: _Msg


@dataclass(frozen=True)
class _Resp:
    choices: list[_Choice]


class _FakeLLM:
    async def chat(self, *, messages: list[dict[str, Any]], temperature: float, stream: bool = False) -> Any:
        assert stream is False
        assert any(m["role"] == "user" for m in messages)
        return _Resp(choices=[_Choice(message=_Msg(content="answer [1]"))])


def test_generate_answer_returns_answer_and_citations(monkeypatch) -> None:
    async def _fake_hybrid_search(**kwargs):
        return [
            _HybridChunk(
                id="a",
                content="fastapi database connection",
                metadata={"source": "doc.md", "chunk_index": 0},
            )
        ]

    def _fake_rerank(**kwargs):
        return [_Reranked(id="a", content="fastapi database connection", metadata={"source": "doc.md"}, score=1.0)]

    monkeypatch.setattr(generator_module, "hybrid_search", _fake_hybrid_search)
    monkeypatch.setattr(generator_module, "rerank", _fake_rerank)
    monkeypatch.setattr(generator_module, "llm_client", _FakeLLM())

    out = asyncio.run(generator_module.generate_answer(query="fastapi database", top_k=1))
    assert out.answer == "answer [1]"
    assert len(out.citations) == 1
    assert out.citations[0].source == "doc.md"
    assert out.citations[0].chunk_index == 0

