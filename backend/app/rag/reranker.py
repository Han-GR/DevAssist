from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from app.rag.bm25 import tokenize


class HasContent(Protocol):
    id: str
    content: str
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class RerankedChunk:
    """
    rerank 之后的 chunk。
    """

    id: str
    content: str
    metadata: dict[str, Any] | None
    score: float


def _overlap_score(*, query: str, content: str) -> float:
    """
    基于关键词重叠的轻量 rerank 分数。

    Args:
        query (str): 查询文本。
        content (str): chunk 内容。

    Returns:
        float: 分数越大表示越相关。

    Notes/Examples:
        这是一个“先把 pipeline 跑通”的占位 reranker：
        - 不依赖额外模型
        - 对专有名词、API 名称、代码符号更敏感
        后续如果要换成 cross-encoder，只需要把这里替换成模型打分即可。
    """
    q = tokenize(query)
    d = tokenize(content)
    if not q or not d:
        return 0.0

    q_set = set(q)
    d_set = set(d)
    hit = len(q_set & d_set)
    return hit / (len(q_set) ** 0.5 + 1.0)


def rerank(
    *, query: str, chunks: Sequence[HasContent], top_k: int = 5
) -> list[RerankedChunk]:
    """
    对候选 chunks 做二次排序（rerank）。

    Args:
        query (str): 用户查询。
        chunks (Sequence[HasContent]): 候选 chunk 列表（向量检索/混合检索的输出）。
        top_k (int): 返回数量，默认 5。

    Returns:
        list[RerankedChunk]: rerank 后的 Top-K 结果。

    Raises:
        ValueError: query 为空或 top_k 非正数时抛出。
    """
    if not query.strip():
        raise ValueError("query is required")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    scored: list[RerankedChunk] = []
    for c in chunks:
        score = _overlap_score(query=query, content=c.content)
        scored.append(RerankedChunk(id=c.id, content=c.content, metadata=c.metadata, score=score))

    scored.sort(key=lambda x: x.score, reverse=True)
    return [s for s in scored[:top_k] if s.score > 0]


def rerank_iter(
    *, query: str, chunks: Iterable[HasContent], top_k: int = 5
) -> list[RerankedChunk]:
    """
    rerank 的 Iterable 版本（便于上游用 generator 供数）。

    Args:
        query (str): 用户查询。
        chunks (Iterable[HasContent]): 候选 chunk 流。
        top_k (int): 返回数量，默认 5。

    Returns:
        list[RerankedChunk]: rerank 后的 Top-K 结果。
    """
    return rerank(query=query, chunks=list(chunks), top_k=top_k)

