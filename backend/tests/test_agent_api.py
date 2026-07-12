from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

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
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def chat(self, *, messages: list[dict[str, Any]], temperature: float, stream: bool = False) -> Any:
        self.seen_messages.append(list(messages))
        _ = temperature
        _ = stream
        if not self._outputs:
            raise RuntimeError("no more outputs")
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=self._outputs.pop(0)))])


def test_agent_returns_json() -> None:
    client = TestClient(main_module.app)
    agent_module.llm_client = _FakeLLMClient(outputs=["Thought: x\nAction: final: ok"])  # type: ignore[assignment]
    persisted: list[dict[str, Any]] = []

    async def _fake_persist_agent_trace_to_db(
        *,
        run_id,
        agent_type: str,
        steps: list[dict[str, Any]],
        result: str | None,
        error: str | None,
        conversation_id=None,
    ) -> None:
        persisted.append(
            {
                "run_id": str(run_id),
                "agent_type": agent_type,
                "steps": steps,
                "result": result,
                "error": error,
                "conversation_id": str(conversation_id) if conversation_id else None,
            }
        )

    agent_module.persist_agent_trace_to_db = _fake_persist_agent_trace_to_db  # type: ignore[assignment]

    resp = client.post("/agent", json={"message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "ok"
    assert body["run_id"]
    assert len(body["steps"]) == 1
    assert persisted and persisted[0]["result"] == "ok"


def test_agent_streaming_returns_sse() -> None:
    client = TestClient(main_module.app)
    agent_module.llm_client = _FakeLLMClient(outputs=["Thought: x\nAction: final: ok"])  # type: ignore[assignment]
    persisted: list[dict[str, Any]] = []

    async def _fake_persist_agent_trace_to_db(
        *,
        run_id,
        agent_type: str,
        steps: list[dict[str, Any]],
        result: str | None,
        error: str | None,
        conversation_id=None,
    ) -> None:
        persisted.append(
            {
                "run_id": str(run_id),
                "agent_type": agent_type,
                "steps": steps,
                "result": result,
                "error": error,
                "conversation_id": str(conversation_id) if conversation_id else None,
            }
        )

    agent_module.persist_agent_trace_to_db = _fake_persist_agent_trace_to_db  # type: ignore[assignment]

    resp = client.post("/agent?stream=true", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert "\"type\": \"meta\"" in text
    assert "\"type\": \"final\"" in text
    assert "\"type\": \"done\"" in text
    assert persisted and persisted[0]["result"] == "ok"


def test_agent_conversation_memory_injected() -> None:
    client = TestClient(main_module.app)
    conversation_id = uuid4()

    asyncio.run(agent_module.short_term_memory.clear())

    fake_llm = _FakeLLMClient(
        outputs=[
            "Thought: x\nAction: final: ok1",
            "Thought: x\nAction: final: ok2",
        ]
    )
    agent_module.llm_client = fake_llm  # type: ignore[assignment]

    async def _fake_persist_agent_trace_to_db(
        *,
        run_id,
        agent_type: str,
        steps: list[dict[str, Any]],
        result: str | None,
        error: str | None,
        conversation_id=None,
    ) -> None:
        _ = run_id
        _ = agent_type
        _ = steps
        _ = result
        _ = error
        _ = conversation_id

    agent_module.persist_agent_trace_to_db = _fake_persist_agent_trace_to_db  # type: ignore[assignment]

    resp1 = client.post("/agent", json={"message": "hi1", "conversation_id": str(conversation_id)})
    assert resp1.status_code == 200
    assert resp1.json()["answer"] == "ok1"

    resp2 = client.post("/agent", json={"message": "hi2", "conversation_id": str(conversation_id)})
    assert resp2.status_code == 200
    assert resp2.json()["answer"] == "ok2"

    assert len(fake_llm.seen_messages) >= 2
    second_call_messages = fake_llm.seen_messages[1]
    assert {"role": "user", "content": "hi1"} in second_call_messages
    assert {"role": "assistant", "content": "ok1"} in second_call_messages
    assert {"role": "user", "content": "hi2"} in second_call_messages
