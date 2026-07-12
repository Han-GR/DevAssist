from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.react import ReActAgent
from app.agent.tools import Tool, ToolRegistry
from app.core.errors import AppError


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, *, messages: list[dict[str, Any]], temperature: float, stream: bool = False) -> Any:
        self.calls.append({"messages": messages, "temperature": temperature, "stream": stream})
        if not self._outputs:
            raise RuntimeError("no more outputs")
        return _FakeResponse(self._outputs.pop(0))


def test_react_agent_multi_tools_then_final() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="search_docs",
            description="search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda *, query, top_k=5: {"results": [{"id": "1", "content": f"doc for {query}"}]},
        )
    )
    registry.register(
        Tool(
            name="execute_code",
            description="exec",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            handler=lambda *, code: {"stdout": "2\n", "stderr": "", "exit_code": 0, "duration_ms": 10},
        )
    )

    llm = _FakeLLM(
        outputs=[
            "Thought: 先查资料\nAction: tool:search_docs\nargs: {\"query\":\"fastapi\", \"top_k\": 2}",
            "Thought: 再运行验证\nAction: tool:execute_code\nargs: {\"code\":\"print(1+1)\"}",
            "Thought: 给结论\nAction: final: ok",
        ]
    )

    agent = ReActAgent(llm=llm, tools=registry, max_iterations=5)  # type: ignore[arg-type]
    final, steps = asyncio.run(agent.run(user_input="x"))

    assert final == "ok"
    assert [s.tool_name for s in steps] == ["search_docs", "execute_code", None]
    assert steps[0].observation["results"][0]["id"] == "1"
    assert steps[1].observation["exit_code"] == 0


def test_react_agent_parses_args_in_code_fence() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda *, text: {"echo": text},
        )
    )

    llm = _FakeLLM(
        outputs=[
            "Thought: x\nAction: tool:echo\nargs: ```json\n{\"text\":\"hi\"}\n```",
            "Thought: y\nAction: final: done",
        ]
    )

    agent = ReActAgent(llm=llm, tools=registry, max_iterations=3)  # type: ignore[arg-type]
    final, steps = asyncio.run(agent.run(user_input="x"))
    assert final == "done"
    assert steps[0].tool_name == "echo"
    assert steps[0].observation == {"echo": "hi"}


def test_react_agent_max_iterations_raises() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="noop",
            description="noop",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            handler=lambda: None,
        )
    )

    llm = _FakeLLM(outputs=["Thought: loop\nAction: tool:noop\nargs: {}"] * 3)
    agent = ReActAgent(llm=llm, tools=registry, max_iterations=2)  # type: ignore[arg-type]

    with pytest.raises(AppError) as exc:
        asyncio.run(agent.run(user_input="x"))
    assert exc.value.code == "agent_max_iterations"

