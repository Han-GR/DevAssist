from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.api.chat as chat_module
import app.main as main_module


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


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeLLMClient:
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        model: str | None = None,
        stream: bool = False,
    ) -> Any:
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="ok"))])


class _FakeRateLimiter:
    def __init__(self, *, allowed: bool) -> None:
        self.allowed = allowed
        self.last_identity: str | None = None

    async def check(self, *, identity: str):
        self.last_identity = identity
        return type(
            "_Decision",
            (),
            {
                "allowed": self.allowed,
                "count": 30,
                "limit": 30,
                "window_seconds": 60,
            },
        )()


def test_rate_limit_rejects_with_429(monkeypatch) -> None:
    chat_module.llm_client = _FakeLLMClient()
    limiter = _FakeRateLimiter(allowed=False)
    monkeypatch.setattr(main_module, "rate_limiter", limiter)

    client = TestClient(main_module.app)
    resp = client.post("/chat", json={"message": "hello", "history": []}, headers={"x-user-id": "u1"})
    assert resp.status_code == 429
    assert resp.headers.get("x-request-id")
    assert resp.headers.get("retry-after") == "60"

    body = resp.json()
    assert body["error"]["code"] == "rate_limited"
    assert body.get("request_id")
    assert limiter.last_identity == "u1"


def test_rate_limit_allows_request(monkeypatch) -> None:
    chat_module.llm_client = _FakeLLMClient()
    limiter = _FakeRateLimiter(allowed=True)
    monkeypatch.setattr(main_module, "rate_limiter", limiter)

    client = TestClient(main_module.app)
    resp = client.post("/chat", json={"message": "hello", "history": []}, headers={"x-user-id": "u2"})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id")
    assert limiter.last_identity == "u2"
