from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.agent as agent_module


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
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    async def chat(self, *, messages: list[dict[str, Any]], temperature: float, stream: bool = False) -> Any:
        _ = messages
        _ = temperature
        _ = stream
        if not self._outputs:
            raise RuntimeError("no more outputs")
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=self._outputs.pop(0)))])


def test_agent_returns_json() -> None:
    client = TestClient(main_module.app)
    agent_module.llm_client = _FakeLLMClient(outputs=["Thought: x\nAction: final: ok"])  # type: ignore[assignment]

    resp = client.post("/agent", json={"message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "ok"
    assert body["run_id"]
    assert len(body["steps"]) == 1


def test_agent_streaming_returns_sse() -> None:
    client = TestClient(main_module.app)
    agent_module.llm_client = _FakeLLMClient(outputs=["Thought: x\nAction: final: ok"])  # type: ignore[assignment]

    resp = client.post("/agent?stream=true", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert "\"type\": \"meta\"" in text
    assert "\"type\": \"final\"" in text
    assert "\"type\": \"done\"" in text

