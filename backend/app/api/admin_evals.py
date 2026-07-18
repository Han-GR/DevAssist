from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, select

from app.db.models import EvalResult
from app.db.session import SessionLocal


router = APIRouter(prefix="/admin")


class EvalResultItem(BaseModel):
    """
    评测结果列表项（管理端用）。
    """

    id: UUID
    eval_type: str
    model_key: str
    metric_name: str
    scope: str
    score: float
    meta: dict[str, Any] | None
    created_at: str


async def list_eval_results_from_db(
    *,
    limit: int = 200,
    eval_type: str | None = None,
    model_key: str | None = None,
    metric_name: str | None = None,
    scope: str | None = None,
) -> list[EvalResultItem]:
    """
    从数据库读取最近的评测结果（按创建时间倒序）。

    Args:
        limit (int): 最大返回条数，默认 200。
        eval_type (str | None): 过滤 eval_type（可选）。
        model_key (str | None): 过滤 model_key（可选）。
        metric_name (str | None): 过滤 metric_name（可选）。
        scope (str | None): 过滤 scope（可选）。

    Returns:
        list[EvalResultItem]: 评测结果列表。

    Raises:
        Exception: 数据库连接或查询失败时原样抛出。

    Notes/Examples:
        - GET /admin/eval-results?eval_type=finetune_rubric&metric_name=pass_rate
    """
    if limit <= 0:
        return []

    stmt = select(EvalResult).order_by(desc(EvalResult.created_at)).limit(limit)
    if eval_type:
        stmt = stmt.where(EvalResult.eval_type == eval_type)
    if model_key:
        stmt = stmt.where(EvalResult.model_key == model_key)
    if metric_name:
        stmt = stmt.where(EvalResult.metric_name == metric_name)
    if scope:
        stmt = stmt.where(EvalResult.scope == scope)

    async with SessionLocal() as session:
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [
            EvalResultItem(
                id=r.id,
                eval_type=r.eval_type,
                model_key=r.model_key,
                metric_name=r.metric_name,
                scope=r.scope,
                score=float(r.score),
                meta=dict(r.meta) if r.meta else None,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]


@router.get("/eval-results", response_model=list[EvalResultItem])
async def list_eval_results(
    limit: int = 200,
    eval_type: str | None = None,
    model_key: str | None = None,
    metric_name: str | None = None,
    scope: str | None = None,
):
    """
    管理端：查看最近的评测结果。

    Args:
        limit (int): 返回条数，默认 200。
        eval_type (str | None): 过滤 eval_type（可选）。
        model_key (str | None): 过滤 model_key（可选）。
        metric_name (str | None): 过滤 metric_name（可选）。
        scope (str | None): 过滤 scope（可选）。

    Returns:
        list[EvalResultItem]: 评测结果列表。

    Notes/Examples:
        - GET /admin/eval-results?limit=200
    """
    return await list_eval_results_from_db(
        limit=limit,
        eval_type=eval_type,
        model_key=model_key,
        metric_name=metric_name,
        scope=scope,
    )


async def get_eval_result_from_db(*, eval_id: UUID) -> EvalResultItem | None:
    """
    从数据库按 id 查询单条评测结果。

    Args:
        eval_id (UUID): 评测结果 id。

    Returns:
        EvalResultItem | None: 找到则返回详情，否则返回 None。

    Raises:
        Exception: 数据库连接或查询失败时原样抛出。
    """
    async with SessionLocal() as session:
        result = await session.execute(select(EvalResult).where(EvalResult.id == eval_id))
        row = result.scalars().first()
        if row is None:
            return None
        return EvalResultItem(
            id=row.id,
            eval_type=row.eval_type,
            model_key=row.model_key,
            metric_name=row.metric_name,
            scope=row.scope,
            score=float(row.score),
            meta=dict(row.meta) if row.meta else None,
            created_at=row.created_at.isoformat(),
        )


@router.get("/eval-results/{eval_id}", response_model=EvalResultItem)
async def get_eval_result(eval_id: UUID):
    """
    管理端：查看单条评测结果详情。

    Args:
        eval_id (UUID): 评测结果 id（路径参数）。

    Returns:
        EvalResultItem: 单条评测结果。

    Raises:
        HTTPException 404: eval_id 不存在时抛出。
    """
    item = await get_eval_result_from_db(eval_id=eval_id)
    if item is None:
        raise HTTPException(status_code=404, detail="eval result not found")
    return item
