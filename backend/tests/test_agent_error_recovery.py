"""
验证：
1. 工具首次失败后会 retry，retry 成功则正常返回 observation。
2. 全部 retry 耗尽后，错误作为 Observation 注入循环（不抛异常），
   模型可以继续推理并给出 final answer。
3. _format_tool_error_observation 格式正确。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.react import (
    TOOL_MAX_RETRIES,
    ReActAgent,
    _call_tool_with_retry,
    _format_tool_error_observation,
)
from app.agent.tools import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# 辅助：构造一个总是失败 N 次、之后成功的 fake tool
# ---------------------------------------------------------------------------

def _make_flaky_registry(fail_times: int) -> tuple[ToolRegistry, list[int]]:
    """返回 (registry, call_count_list)；call_count_list[0] 记录调用次数。"""
    call_count = [0]

    async def _handler(**kwargs: Any) -> dict[str, Any]:
        call_count[0] += 1
        if call_count[0] <= fail_times:
            raise RuntimeError(f"simulated failure #{call_count[0]}")
        return {"ok": True}

    tool = Tool(
        name="flaky_tool",
        description="flaky",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_handler,
    )
    registry = ToolRegistry()
    registry.register(tool)
    return registry, call_count


# ---------------------------------------------------------------------------
# 1. _call_tool_with_retry：首次失败后 retry 成功
# ---------------------------------------------------------------------------

def test_call_tool_with_retry_succeeds_on_second_attempt() -> None:
    registry, call_count = _make_flaky_registry(fail_times=1)
    logger = MagicMock()

    result, error = asyncio.run(
        _call_tool_with_retry(
            tools=registry,
            tool_name="flaky_tool",
            tool_args={},
            max_retries=TOOL_MAX_RETRIES,
            logger=logger,
        )
    )

    assert error is None
    assert result == {"ok": True}
    assert call_count[0] == 2  # 第 1 次失败，第 2 次成功


# ---------------------------------------------------------------------------
# 2. _call_tool_with_retry：全部 retry 耗尽，返回 error 而不抛异常
# ---------------------------------------------------------------------------

def test_call_tool_with_retry_exhausted_returns_error() -> None:
    registry, call_count = _make_flaky_registry(fail_times=TOOL_MAX_RETRIES + 10)
    logger = MagicMock()

    result, error = asyncio.run(
        _call_tool_with_retry(
            tools=registry,
            tool_name="flaky_tool",
            tool_args={},
            max_retries=TOOL_MAX_RETRIES,
            logger=logger,
        )
    )

    assert result is None
    assert error is not None
    assert "simulated failure" in error
    # 调用次数 = 1 首次 + max_retries 次重试
    assert call_count[0] == TOOL_MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# 3. _format_tool_error_observation 格式
# ---------------------------------------------------------------------------

def test_format_tool_error_observation_structure() -> None:
    text = _format_tool_error_observation(
        tool_name="search_docs",
        tool_args={"query": "fastapi"},
        error="connection timeout",
    )
    assert text.startswith("Observation:")
    payload = json.loads(text.split("Observation:\n", 1)[1])
    assert payload["tool_name"] == "search_docs"
    assert payload["error"] == "connection timeout"
    assert "suggestion" in payload


# ---------------------------------------------------------------------------
# 4. ReActAgent 全链路：工具全部失败后模型收到 error observation 并给出 final answer
# ---------------------------------------------------------------------------

def test_react_agent_graceful_degradation_on_tool_failure() -> None:
    """工具全部 retry 失败后，错误作为 Observation 注入，模型给出 final answer。"""
    # 构造一个永远失败的工具
    registry, _ = _make_flaky_registry(fail_times=999)

    # Fake LLM：
    # 第 1 次调用 → 输出 tool call
    # 第 2 次调用（收到 error observation 后）→ 输出 final answer
    call_seq = [
        "Thought: I'll search for info.\nAction: tool:flaky_tool\nargs: {}",
        "Thought: The tool failed, I'll answer from memory.\nAction: final: Sorry, tool unavailable.",
    ]
    call_idx = [0]

    fake_choice = MagicMock()
    fake_choice.message.content = call_seq[0]
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]

    async def _fake_chat(**kwargs: Any) -> Any:
        fake_choice.message.content = call_seq[call_idx[0]]
        call_idx[0] += 1
        return fake_resp

    fake_llm = MagicMock()
    fake_llm.chat = AsyncMock(side_effect=_fake_chat)

    agent = ReActAgent(llm=fake_llm, tools=registry, max_iterations=10)
    answer, steps = asyncio.run(agent.run(user_input="test graceful degradation"))

    assert "Sorry" in answer or "unavailable" in answer
    # 应该有一步记录了 tool_error
    tool_steps = [s for s in steps if s.tool_name == "flaky_tool"]
    assert len(tool_steps) >= 1
