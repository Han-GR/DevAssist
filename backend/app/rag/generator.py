from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.core.llm import LLMClient
from app.rag.reranker import rerank
from app.rag.retriever import HybridChunk, hybrid_search


llm_client: LLMClient | None = None


@dataclass(frozen=True)
class Citation:
    source: str
    chunk_index: int | None
    content: str


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]


def _extract_citations(*, chunks: list[HybridChunk], max_snippet_chars: int = 280) -> list[Citation]:
    """
    从检索 chunk 中提取 citation 列表。

    Args:
        chunks (list[HybridChunk]): 检索/混合检索输出的 chunk 列表。
        max_snippet_chars (int): 每条 citation 的 snippet 最大长度。

    Returns:
        list[Citation]: citations 列表。

    Notes/Examples:
        citation 的 content 只保留一个可读的片段，避免把整段长上下文塞进响应体。
    """
    citations: list[Citation] = []
    for c in chunks:
        meta = c.metadata or {}
        source = str(meta.get("source") or "")
        chunk_index = meta.get("chunk_index")
        idx = int(chunk_index) if isinstance(chunk_index, int) else None
        snippet = c.content.strip().replace("\r\n", "\n")
        citations.append(Citation(source=source, chunk_index=idx, content=snippet[:max_snippet_chars]))
    return citations


def _build_context(*, chunks: list[HybridChunk], max_chars: int = 6000) -> str:
    """
    构建给 LLM 的检索上下文文本。

    Args:
        chunks (list[HybridChunk]): 候选 chunk 列表。
        max_chars (int): 上下文最大字符数（粗略截断）。

    Returns:
        str: 拼接后的上下文文本。
    """
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata or {}
        source = str(meta.get("source") or "")
        chunk_index = meta.get("chunk_index")
        suffix = f"#{chunk_index}" if isinstance(chunk_index, int) else ""
        header = f"[{i}] {source}{suffix}".strip()
        body = c.content.strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain <= 0:
                break
            parts.append(block[:remain])
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts).strip()


async def generate_answer(
    *,
    query: str,
    top_k: int = 5,
    collection_name: str | None = None,
    candidate_multiplier: int = 4,
    rerank_min_score: float = 0.0,
) -> RAGAnswer:
    """
    基于知识库生成带引用的回答（retrieve -> rerank -> prompt -> cite）。

    Args:
        query (str): 用户问题。
        top_k (int): 最终用于生成的引用 chunk 数量，默认 5。
        collection_name (str | None): 指定 collection；不传则使用 Settings 默认值。
        candidate_multiplier (int): rerank 前的候选扩展倍数，默认 4。
        rerank_min_score (float): rerank 最小分数阈值，默认 0.0（等价于旧行为：只保留 score > 0）。

    Returns:
        RAGAnswer: 包含 answer 与 citations（source + snippet）。

    Raises:
        ValueError: query 为空或 top_k 非正数时抛出。
        ConfigurationError: LLM 配置缺失或初始化失败时抛出。
        Exception: embedding/chroma/LLM 调用失败时可能抛出异常（由上层统一处理）。

    Notes/Examples:
        - citations 的编号与上下文编号一致，但当前不强制解析模型输出里的引用标记。
        - 这一步先把“可跑通的闭环”做出来，后续再逐步增强引用格式与严格性。
    """
    if not query.strip():
        raise ValueError("query is required")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be a positive integer")

    settings = get_settings()

    global llm_client
    if llm_client is None:
        try:
            llm_client = LLMClient.from_settings(settings)
        except ValueError as exc:
            raise ConfigurationError(message=str(exc)) from exc

    candidate_k = max(top_k * candidate_multiplier, top_k)
    candidates = await hybrid_search(query=query, top_k=candidate_k, collection_name=collection_name)
    picked = rerank(query=query, chunks=candidates, top_k=top_k, min_score=rerank_min_score)
    picked_by_id = {p.id: p for p in picked}
    final_chunks: list[HybridChunk] = [c for c in candidates if c.id in picked_by_id]

    context = _build_context(chunks=final_chunks)
    system_prompt = (
        "你是一个严谨的技术助手。你只能基于提供的资料回答，不要编造。"
        "回答中需要在相关句子末尾用 [1] [2] 这样的编号标注引用，编号对应资料块的编号。"
        "如果资料不足以回答，就明确说明不知道。"
    )
    user_prompt = f"问题：{query}\n\n资料：\n{context}\n\n请给出回答："

    response = await llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        stream=False,
    )
    answer = response.choices[0].message.content if response.choices else ""
    citations = _extract_citations(chunks=final_chunks)
    return RAGAnswer(answer=answer or "", citations=citations)
