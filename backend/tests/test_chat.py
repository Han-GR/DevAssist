from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.core import config as config_module
import app.main as main_module


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
        self, *, messages: list[dict[str, Any]], temperature: float, stream: bool = False
    ) -> _FakeResponse:
        user_text = messages[0]["content"] if messages else ""
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=f"echo: {user_text}"))])


def test_health_has_request_id() -> None:
    client = TestClient(main_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers.get("x-request-id")


def test_chat_returns_reply() -> None:
    client = TestClient(main_module.app)
    main_module.llm_client = _FakeLLMClient()

    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "echo: hello"


def test_chat_validation_error_is_unified_json() -> None:
    client = TestClient(main_module.app)

    resp = client.post("/chat", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body.get("request_id")


def test_chat_configuration_error_is_unified_json(monkeypatch) -> None:
    client = TestClient(main_module.app)
    main_module.llm_client = None

    def _raise(*args, **kwargs):
        raise ValueError("LLM api_key is required")

    monkeypatch.setattr(main_module.LLMClient, "from_settings", _raise)

    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "configuration_error"
    assert body.get("request_id")


def test_config_loading_from_env(monkeypatch) -> None:
    config_module.get_settings.cache_clear()

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SERVICE_NAME", "devassist-test")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

    settings = config_module.get_settings()
    assert settings.env == "test"
    assert settings.log_level == "WARNING"
    assert settings.service_name == "devassist-test"
    assert settings.llm_provider == "deepseek"
    assert settings.llm_api_key == "sk-test"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_base_url == "https://api.deepseek.com/v1"
