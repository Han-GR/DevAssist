from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from app.core.errors import AppError
from app.rag.ingestion import ingest_text_document


router = APIRouter()


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    collection: str
    chunk_count: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    上传并写入知识库（最小版本）。

    Args:
        file (UploadFile): 上传文件（当前阶段支持 UTF-8 的 .txt/.md/.py/.js）。

    Returns:
        IngestResponse: 写入结果（chunk 数量与 collection 名）。

    Raises:
        AppError: 文件类型不支持、文本解码失败、切分/embedding 结果不符合预期等。
        ConfigurationError: embedding/chroma 配置不完整或不合法。

    Notes/Examples:
        - 这是 ingestion 的最小闭环：upload -> chunk -> embed -> store。
        - 会把文档元信息写入 PostgreSQL 的 documents 表，便于后续做管理与回溯。
    """
    filename = file.filename or "upload"
    lowered = filename.lower()
    if not (
        lowered.endswith(".txt")
        or lowered.endswith(".md")
        or lowered.endswith(".py")
        or lowered.endswith(".js")
    ):
        raise AppError(
            code="unsupported_file_type",
            message="Only .txt, .md, .py and .js are supported for now.",
            status_code=400,
            details={"filename": filename},
        )

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AppError(
            code="invalid_text_encoding",
            message="File must be UTF-8 encoded.",
            status_code=400,
            details={"filename": filename},
        ) from exc

    document_id, chunk_count, collection = await ingest_text_document(
        title=filename,
        source=filename,
        text=text,
    )

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        collection=collection,
        chunk_count=chunk_count,
    )
