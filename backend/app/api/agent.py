from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.agent.builtin_tools import create_execute_code_tool, create_search_docs_tool
from app.agent.react import ReActAgent
from app.agent.tools import ToolRegistry
from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.core.llm import LLMClient
from app.core.streaming import sse_event


settings = get_settings()
router = APIRouter()

llm_client: LLMClient | None = None


class AgentRequest(BaseModel):
    """
    /agent 请求体。

    - message: 用户输入
    - tools: 可选；限制本次 Agent 可用工具集合
    """

    message: str
    tools: list[str] | None = None


class AgentStep(BaseModel):
    """
    单步输出（用于非流式返回）。
    """

    thought: str
    action_raw: str
    tool_name: str | None
    tool_args: dict[str, Any] | None
    observation: Any | None


class AgentResponse(BaseModel):
    """
    /agent 非流式响应体。
    """

    run_id: str
    answer: str
    steps: list[AgentStep]


def _get_llm_client() -> LLMClient:
    """
    获取全局复用的 LLMClient。

    Raises:
        ConfigurationError: 当配置缺失导致 LLMClient 初始化失败时抛出。
    """
    global llm_client
    if llm_client is not None:
        return llm_client

    try:
        llm_client = LLMClient.from_settings(settings)
        return llm_client
    except Exception as exc:
        raise ConfigurationError(message=str(exc)) from exc


def _build_registry(*, allowed_tools: list[str] | None) -> ToolRegistry:
    """
    构建本次请求的工具注册表（可按请求限制工具集合）。
    """
    registry = ToolRegistry()

    tool_builders: dict[str, Any] = {
        "search_docs": create_search_docs_tool,
        "execute_code": create_execute_code_tool,
    }

    selected = allowed_tools or list(tool_builders.keys())
    for name in selected:
        builder = tool_builders.get(name)
        if builder is None:
            continue
        registry.register(builder())

    return registry


@router.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest, stream: bool = False):
    """
    Agent 执行入口。

    - stream=false：返回 JSON（answer + steps）
    - stream=true：返回 SSE（meta/step/final/done/error）
    """
    run_id = uuid4().hex
    tools = _build_registry(allowed_tools=request.tools)
    llm = _get_llm_client()
    agent = ReActAgent(llm=llm, tools=tools)

    if not stream:
        answer, steps = await agent.run(user_input=request.message)
        return AgentResponse(
            run_id=run_id,
            answer=answer,
            steps=[
                AgentStep(
                    thought=s.thought,
                    action_raw=s.action_raw,
                    tool_name=s.tool_name,
                    tool_args=s.tool_args,
                    observation=s.observation,
                )
                for s in steps
            ],
        )

    async def _stream() -> AsyncGenerator[str, None]:
        try:
            yield sse_event(
                data={
                    "type": "meta",
                    "run_id": run_id,
                    "tools": [t.name for t in tools.list()],
                }
            )

            answer, steps = await agent.run(user_input=request.message)
            for s in steps:
                yield sse_event(
                    data={
                        "type": "step",
                        "thought": s.thought,
                        "action_raw": s.action_raw,
                        "tool_name": s.tool_name,
                        "tool_args": s.tool_args,
                        "observation": s.observation,
                    }
                )
            yield sse_event(data={"type": "final", "answer": answer})
            yield sse_event(data={"type": "done"}, event="done")
        except Exception as exc:
            yield sse_event(
                data={"type": "error", "message": "agent_error", "detail": str(exc)},
                event="error",
            )

    return StreamingResponse(_stream(), media_type="text/event-stream")

