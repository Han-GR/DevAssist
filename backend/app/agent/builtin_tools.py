from __future__ import annotations

from typing import Any

import structlog

from app.agent.tools import Tool
from app.core.errors import AppError
from app.agent.sandbox import execute_python
from app.rag.retriever import HybridChunk, hybrid_search


SEARCH_DOCS_SNIPPET_CHARS = 800


def create_search_docs_tool() -> Tool:
    """
    创建 search_docs 工具定义。

    Args:
        None

    Returns:
        Tool: 可注册到 ToolRegistry 的工具实例。

    Raises:
        None

    Notes/Examples:
        该工具用于让 Agent 通过 RAG 检索知识库，并返回结构化的 chunk 列表，供后续推理与回答生成使用。
    """

    async def _handler(
        *, query: str, top_k: int = 5, collection_name: str | None = None
    ) -> dict[str, Any]:
        """
        在知识库中检索与 query 相关的文档片段。

        Args:
            query (str): 用户查询。
            top_k (int): 返回数量，默认 5。
            collection_name (str | None): 指定检索的 collection；不传则用默认值。

        Returns:
            dict[str, Any]: {"results": [...]} 结构，其中 results 每项包含 source/chunk_index/content 等字段。

        Raises:
            AppError: top_k 非法时抛出。
            Exception: 下游检索异常会原样抛出，由上层统一处理。

        Notes/Examples:
            为了便于 LLM 处理，这里会对 content 做截断，避免把超长 chunk 一次性塞回模型上下文。
        """
        if top_k <= 0:
            raise AppError(
                code="tool_input_invalid",
                message="top_k must be a positive integer.",
                status_code=400,
                details={"top_k": top_k},
            )

        logger = structlog.get_logger()
        logger.info(
            "search_docs_start",
            query=query,
            top_k=top_k,
            collection_name=collection_name,
        )

        chunks = await hybrid_search(query=query, top_k=top_k, collection_name=collection_name)
        results = [_format_chunk(c) for c in chunks]
        logger.info("search_docs_done", count=len(results))
        return {"results": results}

    return Tool(
        name="search_docs",
        description="在知识库中检索与问题相关的资料片段，返回结构化的候选上下文（含来源与片段）。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "description": "返回数量，默认 5"},
                "collection_name": {"type": "string", "description": "可选，指定检索的 collection"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        return_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "source": {"type": "string"},
                            "chunk_index": {"type": "integer"},
                            "content": {"type": "string"},
                            "vector_distance": {"type": "number"},
                            "bm25_score": {"type": "number"},
                        },
                        "required": ["id", "source", "chunk_index", "content"],
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def _format_chunk(chunk: HybridChunk) -> dict[str, Any]:
    """
    将 HybridChunk 格式化为可返回给 LLM 的结构。

    Args:
        chunk (HybridChunk): 检索返回的 chunk。

    Returns:
        dict[str, Any]: 结构化 chunk 信息。

    Raises:
        None
    """
    meta = chunk.metadata or {}
    source = str(meta.get("source") or "")
    chunk_index_raw = meta.get("chunk_index")
    chunk_index = int(chunk_index_raw) if isinstance(chunk_index_raw, int) else 0

    content = chunk.content.strip().replace("\r\n", "\n")
    if len(content) > SEARCH_DOCS_SNIPPET_CHARS:
        content = content[:SEARCH_DOCS_SNIPPET_CHARS]

    vector_distance = float(chunk.vector_distance) if chunk.vector_distance is not None else 0.0
    bm25_score = float(chunk.bm25_score) if chunk.bm25_score is not None else 0.0

    return {
        "id": chunk.id,
        "source": source,
        "chunk_index": chunk_index,
        "content": content,
        "vector_distance": vector_distance,
        "bm25_score": bm25_score,
    }


def create_execute_code_tool() -> Tool:
    """
    创建 execute_code 工具定义。

    Args:
        None

    Returns:
        Tool: 可注册到 ToolRegistry 的工具实例。

    Raises:
        None

    Notes/Examples:
        当前阶段只提供 Python 执行能力，后续如需要多语言，可在 sandbox 层扩展。
    """

    async def _handler(*, code: str, timeout_s: int = 5) -> dict[str, Any]:
        """
        在沙箱中执行 Python 代码并返回 stdout/stderr/exit_code。

        Args:
            code (str): 待执行代码。
            timeout_s (int): 执行超时（秒），默认 5。

        Returns:
            dict[str, Any]: sandbox 执行结果。

        Raises:
            AppError: timeout_s 非法时抛出。
            Exception: 底层执行异常原样抛出，由上层统一处理。
        """
        if timeout_s <= 0:
            raise AppError(
                code="tool_input_invalid",
                message="timeout_s must be a positive integer.",
                status_code=400,
                details={"timeout_s": timeout_s},
            )
        return await execute_python(code=code, timeout_s=timeout_s)

    return Tool(
        name="execute_code",
        description="在隔离的沙箱中执行 Python 代码，返回 stdout/stderr/exit_code 等结果。",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
                "timeout_s": {"type": "integer", "description": "超时时间（秒），默认 5"},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        return_schema={
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
                "duration_ms": {"type": "integer"},
            },
            "required": ["stdout", "stderr", "exit_code", "duration_ms"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
