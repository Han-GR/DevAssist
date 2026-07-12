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


def test_react_agent_tool_then_final() -> None:
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
            "Thought: 我先用工具\nAction: tool:echo\nargs: {\"text\":\"hi\"}",
            "Thought: 我拿到了结果\nAction: final: done",
        ]
    )

    agent = ReActAgent(llm=llm, tools=registry, max_iterations=3)  # type: ignore[arg-type]
    final, steps = asyncio.run(agent.run(user_input="hello"))

    assert final == "done"
    assert len(steps) == 2
    assert steps[0].tool_name == "echo"
    assert steps[0].observation == {"echo": "hi"}
    assert steps[1].tool_name is None


def test_react_agent_args_json_invalid_raises_app_error() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="noop",
            description="noop",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            handler=lambda: None,
        )
    )

    llm = _FakeLLM(outputs=["Thought: x\nAction: tool:noop\nargs: {bad json}"])
    agent = ReActAgent(llm=llm, tools=registry, max_iterations=1)  # type: ignore[arg-type]

    with pytest.raises(AppError) as exc:
        asyncio.run(agent.run(user_input="x"))
    assert exc.value.code == "agent_parse_error"

