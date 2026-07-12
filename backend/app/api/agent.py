from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import structlog

from app.agent.builtin_tools import create_execute_code_tool, create_search_docs_tool
from app.agent.react import ReActAgent
from app.agent.trace import TraceRecorder
from app.agent.tools import ToolRegistry
from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.core.llm import LLMClient
from app.core.streaming import sse_event
from app.db.models import AgentTrace
from app.db.session import SessionLocal


settings = get_settings()
router = APIRouter()
logger = structlog.get_logger()

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


def _jsonify(value: Any) -> Any:
    """
    将任意对象尽量转换为 JSON 友好的结构。

    Args:
        value (Any): 任意对象（可能包含不可序列化的类型）。

    Returns:
        Any: 可被 JSON 序列化的结构（dict/list/str/number/...）。

    Raises:
        None

    Notes/Examples:
        默认用 json.dumps(default=str) 把未知类型降级为字符串，避免因为单个字段写库失败。
    """
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


async def persist_agent_trace_to_db(
    *,
    run_id: UUID,
    agent_type: str,
    steps: list[dict[str, Any]],
    result: str | None,
    error: str | None,
    conversation_id: UUID | None = None,
) -> None:
    """
    将一次 Agent 运行的 trace 写入数据库。

    Args:
        run_id (UUID): 本次运行 ID（用于关联一次请求的所有步骤）。
        agent_type (str): Agent 类型（例如 "react"）。
        steps (list[dict[str, Any]]): 步骤列表（建议使用 TraceRecorder.to_dict()["steps"] 的结构）。
        result (str | None): 最终答案；失败时为 None。
        error (str | None): 错误信息；成功时为 None。
        conversation_id (UUID | None): 可选的会话 ID（如果这次 Agent 挂载在 chat 会话下）。

    Returns:
        None

    Raises:
        Exception: 数据库写入失败时可能抛出异常。
    """
    payload_steps = _jsonify(steps)
    async with SessionLocal() as session:
        session.add(
            AgentTrace(
                run_id=run_id,
                conversation_id=conversation_id,
                agent_type=agent_type,
                steps=payload_steps,
                result=result,
                error=error,
            )
        )
        await session.commit()


async def _safe_persist_agent_trace(
    *,
    run_id: UUID,
    agent_type: str,
    steps: list[dict[str, Any]],
    result: str | None,
    error: str | None,
    conversation_id: UUID | None = None,
) -> None:
    """
    尝试落库 Agent trace（失败不影响主流程）。

    Args:
        run_id (UUID): 本次运行 ID。
        agent_type (str): Agent 类型。
        steps (list[dict[str, Any]]): trace steps。
        result (str | None): 最终答案。
        error (str | None): 错误信息。
        conversation_id (UUID | None): 会话 ID。

    Returns:
        None

    Raises:
        None
    """
    try:
        await persist_agent_trace_to_db(
            run_id=run_id,
            agent_type=agent_type,
            steps=steps,
            result=result,
            error=error,
            conversation_id=conversation_id,
        )
    except Exception as exc:
        logger.error("agent_trace_persist_failed", run_id=str(run_id), error=str(exc))


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
    run_id = uuid4()
    tools = _build_registry(allowed_tools=request.tools)
    llm = _get_llm_client()
    agent = ReActAgent(llm=llm, tools=tools)
    trace = TraceRecorder(run_id=str(run_id))

    if not stream:
        try:
            answer, steps = await agent.run(user_input=request.message, trace=trace)
        except Exception as exc:
            await _safe_persist_agent_trace(
                run_id=run_id,
                agent_type="react",
                steps=trace.to_dict()["steps"],
                result=None,
                error=str(exc),
            )
            raise

        await _safe_persist_agent_trace(
            run_id=run_id,
            agent_type="react",
            steps=trace.to_dict()["steps"],
            result=answer,
            error=None,
        )
        return AgentResponse(
            run_id=str(run_id),
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
                    "run_id": str(run_id),
                    "tools": [t.name for t in tools.list()],
                }
            )

            answer, steps = await agent.run(user_input=request.message, trace=trace)
            await _safe_persist_agent_trace(
                run_id=run_id,
                agent_type="react",
                steps=trace.to_dict()["steps"],
                result=answer,
                error=None,
            )
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
            await _safe_persist_agent_trace(
                run_id=run_id,
                agent_type="react",
                steps=trace.to_dict()["steps"],
                result=None,
                error=str(exc),
            )
            yield sse_event(
                data={"type": "error", "message": "agent_error", "detail": str(exc)},
                event="error",
            )

    return StreamingResponse(_stream(), media_type="text/event-stream")
