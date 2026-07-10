from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.chat as chat_module
import app.rag.generator as generator_module


@pytest.fixture(autouse=True)
def _stub_db(monkeypatch) -> None:
    async def _empty_history(*args, **kwargs):
        return []

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(chat_module, "load_history_from_db", _empty_history)
    monkeypatch.setattr(chat_module, "persist_user_message_to_db", _noop)
    monkeypatch.setattr(chat_module, "persist_assistant_message_to_db", _noop)
    monkeypatch.setattr(chat_module, "persist_turn_to_db", _noop)


def test_chat_use_rag_returns_formatted_reply(monkeypatch) -> None:
    async def _fake_generate_answer(**kwargs):
        return generator_module.RAGAnswer(
            answer="rag answer [1]",
            citations=[generator_module.Citation(source="doc.md", chunk_index=0, content="snippet")],
        )

    monkeypatch.setattr(chat_module.rag_generator, "generate_answer", _fake_generate_answer)

    client = TestClient(main_module.app)
    resp = client.post("/chat", json={"message": "what is fastapi", "use_rag": True})
    assert resp.status_code == 200
    body = resp.json()
    assert "rag answer" in body["reply"]
    assert "Sources:" in body["reply"]
    assert "doc.md#0" in body["reply"]


def test_chat_use_rag_streaming_returns_sse(monkeypatch) -> None:
    async def _fake_generate_answer(**kwargs):
        return generator_module.RAGAnswer(
            answer="rag answer [1]",
            citations=[generator_module.Citation(source="doc.md", chunk_index=0, content="snippet")],
        )

    monkeypatch.setattr(chat_module.rag_generator, "generate_answer", _fake_generate_answer)

    client = TestClient(main_module.app)
    resp = client.post("/chat?stream=true", json={"message": "what is fastapi", "use_rag": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "\"rag\": true" in resp.text
    assert "rag answer" in resp.text

