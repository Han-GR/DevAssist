from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.agent.memory import MemoryManager, ShortTermMemory


class _FakeLongTermMemory:
    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []
        self.to_return: list[str] = []

    async def add_summary(self, *, conversation_id, summary: str) -> None:
        self.added.append({"conversation_id": str(conversation_id), "summary": summary})

    async def search(self, *, conversation_id, query: str, top_k: int = 3) -> list[str]:
        _ = conversation_id
        _ = query
        _ = top_k
        return list(self.to_return)


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeLLM:
    def __init__(self, output: str) -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages: list[dict[str, Any]], temperature: float, stream: bool = False) -> Any:
        self.calls.append({"messages": messages, "temperature": temperature, "stream": stream})
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=self._output))])


def test_memory_manager_summarizes_evicted_messages_into_long_term() -> None:
    cid = uuid4()
    short = ShortTermMemory(max_turns=1, max_conversations=10)
    long = _FakeLongTermMemory()
    mgr = MemoryManager(short_term=short, long_term=long, summarize_min_messages=2)  # type: ignore[arg-type]

    llm = _FakeLLM(output="SUMMARY")

    asyncio.run(mgr.add_turn(conversation_id=cid, user="u1", assistant="a1", llm=llm))  # type: ignore[arg-type]
    asyncio.run(mgr.add_turn(conversation_id=cid, user="u2", assistant="a2", llm=llm))  # type: ignore[arg-type]

    assert long.added and long.added[0]["summary"] == "SUMMARY"
    history = asyncio.run(short.get_history(conversation_id=cid))
    assert history == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_memory_manager_build_history_includes_long_term_results() -> None:
    cid = uuid4()
    short = ShortTermMemory(max_turns=1, max_conversations=10)
    long = _FakeLongTermMemory()
    long.to_return = ["A", "B"]
    mgr = MemoryManager(short_term=short, long_term=long, summarize_min_messages=2)  # type: ignore[arg-type]

    history = asyncio.run(mgr.build_history(conversation_id=cid, query="q"))
    assert history and history[0]["role"] == "system"
    assert "Long-term memory" in history[0]["content"]
