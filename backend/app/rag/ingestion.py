from __future__ import annotations

from uuid import UUID, uuid4

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ConfigurationError
from app.db.models import Document
from app.db.session import SessionLocal
from app.rag.chroma import ChromaCollectionManager
from app.rag.embedder import Embedder
from app.rag.splitter import split_text_semantic


embedder: Embedder | None = None
chroma_manager: ChromaCollectionManager | None = None


def get_embedder(*, settings: Settings) -> Embedder:
    """
    获取 Embedder 单例。

    Args:
        settings (Settings): 应用配置对象。

    Returns:
        Embedder: 复用的 embedding 客户端实例。

    Raises:
        ConfigurationError: embedding 配置缺失或不合法时抛出。

    Notes/Examples:
        ingestion 是高频 I/O 路径，复用 client 可以减少重复创建连接的开销。
    """
    global embedder

    if embedder is not None:
        return embedder

    try:
        embedder = Embedder.from_settings(settings)
    except ValueError as exc:
        raise ConfigurationError(message=str(exc)) from exc

    return embedder


def get_chroma_manager(*, settings: Settings) -> ChromaCollectionManager:
    """
    获取 ChromaCollectionManager 单例。

    Args:
        settings (Settings): 应用配置对象。

    Returns:
        ChromaCollectionManager: 复用的 collection 管理器。

    Raises:
        ConfigurationError: chroma 配置不合法时抛出。

    Notes/Examples:
        目前先直连 Chroma HttpClient；后续如果需要更复杂的策略，再在 manager 内扩展。
    """
    global chroma_manager

    if chroma_manager is not None:
        return chroma_manager

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
        当前只做“最小可追踪”，后续做删除/重建索引时可用 document_id 串起 DB 与向量库。
    """
    async with SessionLocal() as session:
        doc = Document(title=title, source=source, chunk_count=chunk_count)
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        return doc.id


async def ingest_text_document(
    *,
    title: str,
    source: str,
    text: str,
    collection_name: str | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
) -> tuple[UUID, int, str]:
    """
    把一段文本写入知识库（chunk -> embed -> store -> persist meta）。

    Args:
        title (str): 文档标题（展示/管理用）。
        source (str): 文档来源标识（用于回溯）。
        text (str): 原始文本内容（UTF-8）。
        collection_name (str | None): 指定写入的 collection；不传则使用 Settings 默认值。
        chunk_size (int): 切分目标长度，默认 512。
        overlap (int): chunk 重叠长度，默认 64。

    Returns:
        tuple[UUID, int, str]:
            - document_id：写入 documents 表后的 id
            - chunk_count：切分后的 chunk 数量
            - collection：实际写入的 collection 名

    Raises:
        AppError: 文档为空、embedding 数量不匹配等业务可预期错误。
        ConfigurationError: embedding/chroma 配置不完整或不合法。
        Exception: 网络/数据库等底层异常会原样抛出，由上层统一处理。

    Notes/Examples:
        这是 ingestion 的“业务核心”，API 与脚本都应复用这里，避免两套逻辑慢慢漂移。
    """
    settings = get_settings()
    collection = collection_name or settings.chroma_collection

    chunks = split_text_semantic(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise AppError(
            code="empty_document",
            message="Document is empty.",
            status_code=400,
            details={"source": source},
        )

    embedding_client = get_embedder(settings=settings)
    vectors = await embedding_client.embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise AppError(
            code="embedding_mismatch",
            message="Embedding results do not match chunks.",
            status_code=500,
            details={"chunk_count": len(chunks), "vector_count": len(vectors)},
        )

    manager = get_chroma_manager(settings=settings)
    chroma_collection = manager.get_or_create_collection(name=collection)

    ids = [uuid4().hex for _ in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    chroma_collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    document_id = await persist_document_to_db(
        title=title,
        source=source,
        chunk_count=len(chunks),
    )

    return document_id, len(chunks), collection
