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


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))


def test_http_request_log_contains_duration_and_user_id(monkeypatch) -> None:
    chat_module.llm_client = _FakeLLMClient()
    fake_logger = _FakeLogger()
    monkeypatch.setattr(main_module, "logger", fake_logger)

    client = TestClient(main_module.app)
    resp = client.post("/chat", json={"message": "hello"}, headers={"x-user-id": "u1"})
    assert resp.status_code == 200

    http_events = [item for item in fake_logger.events if item[0] == "http_request"]
    assert len(http_events) == 1
    _, fields = http_events[0]
    assert fields["status_code"] == 200
    assert fields["user_id"] == "u1"
    assert isinstance(fields["duration_ms"], int)
    assert fields["duration_ms"] >= 0
