from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import desc, select

from app.db.models import AgentTrace
from app.db.session import SessionLocal


router = APIRouter(prefix="/admin")


class AgentTraceItem(BaseModel):
    """
    Agent trace 列表项（管理端用）。
    """

    run_id: UUID
    conversation_id: UUID | None
    agent_type: str
    steps: list[dict[str, Any]]
    result: str | None
    error: str | None
    created_at: str


async def list_agent_traces_from_db(*, limit: int = 50) -> list[AgentTraceItem]:
    """
    从数据库读取最近的 Agent traces（按创建时间倒序）。

    Args:
        limit (int): 最大返回条数，默认 50。

    Returns:
        list[AgentTraceItem]: trace 列表。

    Raises:
        Exception: 数据库连接或查询失败时原样抛出。

    Notes/Examples:
        - 该函数用于管理端 trace 列表展示，返回结构尽量“开箱即用”。
    """
    if limit <= 0:
        return []

    async with SessionLocal() as session:
        result = await session.execute(
            select(AgentTrace).order_by(desc(AgentTrace.created_at)).limit(limit)
        )
        rows = result.scalars().all()
        return [
            AgentTraceItem(
                run_id=r.run_id,
                conversation_id=r.conversation_id,
                agent_type=r.agent_type,
                steps=list(r.steps or []),
                result=r.result,
                error=r.error,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]


@router.get("/agent-traces", response_model=list[AgentTraceItem])
async def list_agent_traces(limit: int = 50):
    """
    管理端：查看最近的 Agent traces。

    Args:
        limit (int): 返回条数，默认 50。

    Returns:
        list[AgentTraceItem]: trace 列表。

    Raises:
        Exception: 数据库异常会交给全局异常处理器统一处理。

    Notes/Examples:
        - GET /admin/agent-traces?limit=50
    """
    return await list_agent_traces_from_db(limit=limit)
