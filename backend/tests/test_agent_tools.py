from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.tools import Tool, ToolRegistry
from app.core.errors import AppError


def test_tool_schema_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Tool(
            name="bad",
            description="bad",
            parameters={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]},
            handler=lambda **_: None,
        )


def test_registry_register_and_get() -> None:
    registry = ToolRegistry()
    tool = Tool(
        name="add",
        description="add two integers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=lambda *, a, b: a + b,
    )

    registry.register(tool)
    assert registry.get("add").name == "add"


def test_registry_register_duplicate_raises_app_error() -> None:
    registry = ToolRegistry()
    tool = Tool(
        name="t",
        description="x",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=lambda: None,
    )
    registry.register(tool)
    with pytest.raises(AppError) as exc:
        registry.register(tool)
    assert exc.value.code == "tool_already_registered"
    assert exc.value.status_code == 409


def test_registry_get_missing_raises_app_error() -> None:
    registry = ToolRegistry()
    with pytest.raises(AppError) as exc:
        registry.get("missing")
    assert exc.value.code == "tool_not_found"
    assert exc.value.status_code == 404


def test_tool_validate_missing_required_raises_app_error() -> None:
    tool = Tool(
        name="search",
        description="search",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        handler=lambda *, query, top_k=5: {"query": query, "top_k": top_k},
    )
    with pytest.raises(AppError) as exc:
        tool.validate_input({"top_k": 3})
    assert exc.value.code == "tool_input_invalid"


def test_tool_validate_type_mismatch_raises_app_error() -> None:
    tool = Tool(
        name="search",
        description="search",
        parameters={
            "type": "object",
            "properties": {"top_k": {"type": "integer"}},
            "required": ["top_k"],
        },
        handler=lambda *, top_k: top_k,
    )
    with pytest.raises(AppError) as exc:
        tool.validate_input({"top_k": "3"})
    assert exc.value.code == "tool_input_invalid"


def test_tool_call_sync_handler() -> None:
    tool = Tool(
        name="add",
        description="add",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=lambda *, a, b: a + b,
    )
    assert asyncio.run(tool.call({"a": 1, "b": 2})) == 3


def test_tool_call_async_handler() -> None:
    async def handler(*, query: str) -> str:
        return query.upper()

    tool = Tool(
        name="upper",
        description="upper",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=handler,
    )
    assert asyncio.run(tool.call({"query": "ok"})) == "OK"


def test_registry_call() -> None:
    registry = ToolRegistry()
    tool = Tool(
        name="add",
        description="add",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=lambda *, a, b: a + b,
    )
    registry.register(tool)
    assert asyncio.run(registry.call(name="add", payload={"a": 2, "b": 3})) == 5

