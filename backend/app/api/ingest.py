from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.errors import AppError, ConfigurationError
from app.db.models import Document
from app.db.session import SessionLocal
from app.rag.chroma import ChromaCollectionManager
from app.rag.embedder import Embedder
from app.rag.splitter import split_text_semantic


router = APIRouter()

embedder: Embedder | None = None
chroma_manager: ChromaCollectionManager | None = None


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    collection: str
    chunk_count: int


def get_embedder() -> Embedder:
    """
    获取 Embedder 单例。

    Returns:
        Embedder: 复用的 embedding 客户端实例。

    Raises:
        ConfigurationError: embedding 配置缺失或不合法时抛出。

    Notes/Examples:
        ingest 是一个“高频的 I/O 接口”，Embedder 复用能减少底层 client 的重复创建开销。
    """
    global embedder

    if embedder is not None:
        return embedder

    settings = get_settings()
    try:
        embedder = Embedder.from_settings(settings)
    except ValueError as exc:
        raise ConfigurationError(message=str(exc)) from exc

    return embedder


def get_chroma_manager() -> ChromaCollectionManager:
    """
    获取 ChromaCollectionManager 单例。

    Returns:
        ChromaCollectionManager: 复用的 collection 管理器。

    Raises:
        ConfigurationError: chroma 配置不合法时抛出。

    Notes/Examples:
        这里先用 HttpClient 直连 Chroma 服务；后续需要更复杂的 collection 策略，再在 manager 里扩展即可。
    """
    global chroma_manager

    if chroma_manager is not None:
        return chroma_manager

    settings = get_settings()
    try:
        chroma_manager = ChromaCollectionManager.from_settings(settings)
    except ValueError as exc:
        raise ConfigurationError(message=str(exc)) from exc

    return chroma_manager


async def persist_document_to_db(*, title: str, source: str, chunk_count: int) -> UUID:
    """
    把一次 ingestion 的元信息写入 documents 表。

    Args:
        title (str): 文档标题（当前阶段一般直接用文件名）。
        source (str): 文档来源标识（当前阶段一般直接用文件名/路径）。
        chunk_count (int): 切分后的 chunk 数量。

    Returns:
        UUID: 新建的 document_id。

    Raises:
        Exception: 数据库连接或写入失败时可能抛出异常（由全局异常处理器统一处理）。

    Notes/Examples:
        目前只写“最小可追踪”的字段，后续做删除/重建索引时，可以通过 document_id 反查对应的向量条目。
    """
    async with SessionLocal() as session:
        doc = Document(title=title, source=source, chunk_count=chunk_count)
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        return doc.id


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    上传并写入知识库（最小版本）。

    Args:
        file (UploadFile): 上传文件（当前阶段只支持 UTF-8 的 .txt/.md）。

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
    if not (lowered.endswith(".txt") or lowered.endswith(".md")):
        raise AppError(
            code="unsupported_file_type",
            message="Only .txt and .md are supported for now.",
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

    chunks = split_text_semantic(text, chunk_size=512, overlap=64)
    if not chunks:
        raise AppError(
            code="empty_document",
            message="File is empty.",
            status_code=400,
            details={"filename": filename},
        )

    embedding_client = get_embedder()
    vectors = await embedding_client.embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise AppError(
            code="embedding_mismatch",
            message="Embedding results do not match chunks.",
            status_code=500,
            details={"chunk_count": len(chunks), "vector_count": len(vectors)},
        )

    settings = get_settings()
    manager = get_chroma_manager()
    collection = manager.get_or_create_collection(name=settings.chroma_collection)

    ids = [uuid4().hex for _ in range(len(chunks))]
    metadatas = [
        {"source": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    document_id = await persist_document_to_db(
        title=filename,
        source=filename,
        chunk_count=len(chunks),
    )

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        collection=settings.chroma_collection,
        chunk_count=len(chunks),
    )
