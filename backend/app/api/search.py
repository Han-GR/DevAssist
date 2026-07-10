from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.errors import AppError
from app.rag.reranker import rerank
from app.rag.retriever import HybridChunk, hybrid_search


router = APIRouter()

DEFAULT_RERANK_CANDIDATE_MULTIPLIER = 4


class SearchRequest(BaseModel):
    """
    /search 请求体。
    """

    query: str
    top_k: int = 5
    collection_name: str | None = None


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] | None
    score: float
    vector_distance: float | None
    bm25_score: float | None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


def _validate_search_request(req: SearchRequest) -> None:
    """
    校验 search 请求参数。

    Args:
        req (SearchRequest): 请求体。

    Returns:
        None

    Raises:
        AppError: 参数不合法时抛出。
    """
    if not req.query.strip():
        raise AppError(code="invalid_query", message="query is required.", status_code=400)
    if req.top_k <= 0:
        raise AppError(code="invalid_top_k", message="top_k must be a positive integer.", status_code=400)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """
    检索接口：hybrid（vector + BM25）-> rerank -> Top-K。

    Args:
        req (SearchRequest): 查询请求。

    Returns:
        SearchResponse: 检索结果列表。

    Raises:
        AppError: 参数不合法，或检索结果不符合预期时抛出。
        Exception: embedding/chroma 等底层异常会原样抛出，由全局异常处理器统一处理。
    """
    _validate_search_request(req)

    candidate_k = max(req.top_k * DEFAULT_RERANK_CANDIDATE_MULTIPLIER, req.top_k)
    candidates = await hybrid_search(query=req.query, top_k=candidate_k, collection_name=req.collection_name)
    by_id: dict[str, HybridChunk] = {c.id: c for c in candidates}

    reranked = rerank(query=req.query, chunks=candidates, top_k=req.top_k)
    results: list[SearchResult] = []
    for r in reranked:
        src = by_id.get(r.id)
        if src is None:
            continue
        results.append(
            SearchResult(
                id=r.id,
                content=r.content,
                metadata=r.metadata,
                score=r.score,
                vector_distance=src.vector_distance,
                bm25_score=src.bm25_score,
            )
        )

    return SearchResponse(query=req.query, results=results)

