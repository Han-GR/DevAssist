"""
聊天接口。

这个路由提供一个统一入口：
- 非流式：返回 JSON（conversation_id + reply）
- 流式：返回 SSE（meta/delta/done）

Day10 开始接入 PostgreSQL：
- 请求带 conversation_id 时优先用 DB 历史
- 消息会写入 messages 表，支持后续做会话列表/回放
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select
from starlette.responses import StreamingResponse

from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.core.llm import LLMClient
from app.core.streaming import sse_event
from app.db.models import Conversation, Message
from app.db.session import SessionLocal
import app.rag.generator as rag_generator


settings = get_settings()
router = APIRouter()

llm_client: LLMClient | None = None


class ChatMessage(BaseModel):
    """
    一条聊天消息（用于请求体 history）。

    这里保持和 OpenAI 的 role/content 形状一致，后续接 RAG 的 system prompt 也能复用。
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """
    /chat 请求体。

    - conversation_id：可选；传入时表示继续某个会话
    - history：仅在未传 conversation_id 时作为上下文使用（兼容早期实现）
    """

    conversation_id: UUID | None = None
    message: str
    history: list[ChatMessage] = []
    use_rag: bool | None = None
    collection_name: str | None = None


class ChatResponse(BaseModel):
    """
    /chat 非流式响应体。

    conversation_id 总会返回，前端可以用它把下一轮对话串起来。
    """

    conversation_id: UUID
    reply: str


def _should_use_rag(*, message: str, force: bool | None) -> bool:
    """
    判断是否启用 RAG。

    Args:
        message (str): 用户输入。
        force (bool | None): 显式开关；不为 None 时直接按该值执行。

    Returns:
        bool: True 表示走 RAG，False 表示走普通聊天。

    Notes/Examples:
        当前策略偏保守：默认只在“看起来像技术问答”时才启用 RAG，避免闲聊也去跑检索。
    """
    if force is not None:
        return force

    if not settings.embedding_model or not settings.embedding_api_key:
        return False

    text = message.strip()
    if not text:
        return False

    lowered = text.lower()
    if "?" in text or "？" in text:
        return True

    triggers = [
        "怎么",
        "如何",
        "为什么",
        "报错",
        "错误",
        "traceback",
        "exception",
        "fastapi",
        "sqlalchemy",
        "alembic",
        "docker",
        "postgres",
        "chroma",
        "embedding",
    ]
    return any(t in lowered for t in triggers)


def _format_rag_reply(*, answer: rag_generator.RAGAnswer) -> str:
    """
    把 RAGAnswer 格式化为适合直接回传给前端的文本。

    Args:
        answer (RAGAnswer): RAG 生成结果（含 citations）。

    Returns:
        str: 拼接后的 Markdown 文本。
    """
    lines: list[str] = [answer.answer.strip()]
    if answer.citations:
        lines.append("")
        lines.append("Sources:")
        for i, c in enumerate(answer.citations, start=1):
            suffix = f"#{c.chunk_index}" if c.chunk_index is not None else ""
            src = f"{c.source}{suffix}".strip()
            lines.append(f"{i}. {src}")
    return "\n".join([x for x in lines if x is not None]).strip()


def _split_to_stream_parts(*, text: str, chunk_chars: int = 200) -> list[str]:
    """
    把一段文本切成更适合 SSE 发送的多个片段。

    Args:
        text (str): 完整文本。
        chunk_chars (int): 每个片段的最大字符数。

    Returns:
        list[str]: 文本片段列表。
    """
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


async def load_history_from_db(conversation_id: UUID) -> list[dict[str, str]]:
    """
    从数据库加载某个会话的历史消息。

    Args:
        conversation_id (UUID): 会话 ID。

    Returns:
        list[dict[str, str]]: 按时间升序排列的历史消息列表，每项为 {"role": "...", "content": "..."}。

    Raises:
        Exception: 数据库连接、查询失败时可能抛出异常（原样上抛给全局异常处理器）。

    Notes:
        按 created_at 升序取消息，保证拼给 LLM 的上下文顺序稳定且可复现，也方便后续做“回放”和排查问题。
    """
    async with SessionLocal() as session:
        result = await session.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        rows = result.all()
        return [{"role": role, "content": content} for role, content in rows]


async def _ensure_conversation(*, session, conversation_id: UUID) -> None:
    """
    确保 conversations 表里存在该 conversation_id。

    Args:
        session: SQLAlchemy AsyncSession（由调用方创建并传入）。
        conversation_id (UUID): 会话 ID。

    Returns:
        None

    Raises:
        Exception: 数据库读写失败时可能抛出异常。

    Notes:
        单独抽这一层，是为了让写消息的函数不用关心“会话是否已创建”，逻辑更清爽。
    """
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        session.add(Conversation(id=conversation_id))
        await session.commit()


async def persist_user_message_to_db(conversation_id: UUID, content: str) -> None:
    """
    把用户消息写入 messages 表。

    Args:
        conversation_id (UUID): 会话 ID。
        content (str): 用户消息内容。

    Returns:
        None

    Raises:
        Exception: 数据库写入失败时可能抛出异常。

    Notes:
        流式场景下会先把 user message 落库，这样即使后面 LLM 流被中断，
        至少用户提问不会丢，便于恢复和排查。
    """
    async with SessionLocal() as session:
        await _ensure_conversation(session=session, conversation_id=conversation_id)
        session.add(
            Message(conversation_id=conversation_id, role="user", content=content)
        )
        await session.commit()


async def persist_assistant_message_to_db(conversation_id: UUID, content: str) -> None:
    """
    把助手消息写入 messages 表。

    Args:
        conversation_id (UUID): 会话 ID。
        content (str): 助手消息内容。

    Returns:
        None

    Raises:
        Exception: 数据库写入失败时可能抛出异常。

    注意：对流式响应来说，我们不会把每个 delta 都写数据库，
    而是等流式结束后拼成完整文本再写入，避免存一堆碎片。
    """
    async with SessionLocal() as session:
        await _ensure_conversation(session=session, conversation_id=conversation_id)
        session.add(
            Message(conversation_id=conversation_id, role="assistant", content=content)
        )
        await session.commit()


async def persist_turn_to_db(conversation_id: UUID, user: str, assistant: str) -> None:
    """
    把一轮对话（user + assistant）一次性写入数据库。

    Args:
        conversation_id (UUID): 会话 ID。
        user (str): 用户消息内容。
        assistant (str): 助手回复内容。

    Returns:
        None

    Raises:
        Exception: 数据库写入失败时可能抛出异常。

    这是给非流式场景用的：拿到完整回复后再统一落库，
    既简单也能保证一问一答在时间上更紧凑。
    """
    async with SessionLocal() as session:
        await _ensure_conversation(session=session, conversation_id=conversation_id)
        session.add(Message(conversation_id=conversation_id, role="user", content=user))
        session.add(
            Message(conversation_id=conversation_id, role="assistant", content=assistant)
        )
        await session.commit()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    stream: bool = False,
) -> ChatResponse | StreamingResponse:
    """
    聊天接口：支持非流式 JSON 返回，也支持 SSE 流式输出。

    Args:
        payload (ChatRequest): 请求体（message/history/conversation_id）。
        stream (bool): 是否启用 SSE 流式输出（来自 query param）。

    Returns:
        ChatResponse | StreamingResponse:
            - stream=false：返回 ChatResponse（conversation_id + reply）
            - stream=true：返回 StreamingResponse（text/event-stream）

    Raises:
        ConfigurationError: LLM 配置缺失、初始化失败时抛出。
        Exception: LLM 调用失败或数据库读写失败时可能抛出异常（由全局异常处理器统一处理）。

    Notes:
        - conversation_id：不传则服务端生成；传入则复用并优先从 DB 读取历史
        - history：仅在未传 conversation_id 时生效（兼容早期纯客户端拼历史）
        - stream=true：先写入 user 消息，流式结束后再写入完整 assistant 消息（避免半截内容入库）
    """
    global llm_client

    if llm_client is None:
        try:
            llm_client = LLMClient.from_settings(settings)
        except ValueError as exc:
            raise ConfigurationError(message=str(exc)) from exc

    rag_generator.llm_client = llm_client

    conversation_id = payload.conversation_id or uuid4()

    if payload.conversation_id is not None:
        messages = await load_history_from_db(conversation_id)
    else:
        messages = [{"role": m.role, "content": m.content} for m in payload.history]
    messages.append({"role": "user", "content": payload.message})

    use_rag = _should_use_rag(message=payload.message, force=payload.use_rag)

    if use_rag:
        if stream:
            await persist_user_message_to_db(conversation_id, payload.message)
            rag_answer = await rag_generator.generate_answer(
                query=payload.message,
                top_k=5,
                collection_name=payload.collection_name,
            )
            reply_text = _format_rag_reply(answer=rag_answer)
            parts = _split_to_stream_parts(text=reply_text)

            async def _rag_generator() -> AsyncGenerator[str, None]:
                yield sse_event(
                    data={"type": "meta", "conversation_id": str(conversation_id), "rag": True}
                )
                for p in parts:
                    if p:
                        yield sse_event(data={"type": "delta", "content": p})
                yield sse_event(data={"type": "done"}, event="done")

            async def _persist_after_rag_stream() -> AsyncGenerator[str, None]:
                try:
                    async for item in _rag_generator():
                        yield item
                finally:
                    if reply_text:
                        await persist_assistant_message_to_db(conversation_id, reply_text)

            return StreamingResponse(
                _persist_after_rag_stream(), media_type="text/event-stream"
            )

        rag_answer = await rag_generator.generate_answer(
            query=payload.message,
            top_k=5,
            collection_name=payload.collection_name,
        )
        reply_text = _format_rag_reply(answer=rag_answer)
        await persist_turn_to_db(conversation_id, payload.message, reply_text)
        return ChatResponse(conversation_id=conversation_id, reply=reply_text)

    if stream:
        await persist_user_message_to_db(conversation_id, payload.message)
        openai_stream = await llm_client.chat(
            messages=messages,
            temperature=0.0,
            stream=True,
        )
        assistant_parts: list[str] = []

        async def _generator() -> AsyncGenerator[str, None]:
            """
            将 OpenAI-style 流转换成 SSE 事件流。

            Yields:
                str: SSE 文本片段（meta/delta/done）。

            Raises:
                Exception: 迭代 openai_stream 时可能抛出异常（由外层包装处理）。

            Notes:
                - 先发 meta（把 conversation_id 告诉前端，方便前端立即保存）
                - 再持续发 delta（每个 chunk 的增量内容）
                - 最后发 done（让前端收尾）
            """
            yield sse_event(
                data={"type": "meta", "conversation_id": str(conversation_id)}
            )

            async for chunk in openai_stream:
                delta = None
                try:
                    delta = chunk.choices[0].delta
                except Exception:
                    delta = None

                content = (
                    getattr(delta, "content", None) if delta is not None else None
                )
                if content:
                    assistant_parts.append(content)
                    yield sse_event(data={"type": "delta", "content": content})

            yield sse_event(data={"type": "done"}, event="done")

        async def _persist_after_stream() -> AsyncGenerator[str, None]:
            """
            包一层 finally：保证流式结束后把完整 assistant 内容写入数据库。

            Yields:
                str: SSE 文本片段（转发自 _generator）。

            Raises:
                Exception: finally 里写数据库失败时可能抛出异常。

            Notes:
                这么做是为了避免异常、断连等情况下漏写数据库，同时也避免把 delta 碎片化存储。
            """
            try:
                async for item in _generator():
                    yield item
            finally:
                assistant_text = "".join(assistant_parts)
                if assistant_text:
                    await persist_assistant_message_to_db(conversation_id, assistant_text)

        return StreamingResponse(_persist_after_stream(), media_type="text/event-stream")

    response = await llm_client.chat(
        messages=messages,
        temperature=0.0,
        stream=False,
    )
    content = response.choices[0].message.content if response.choices else ""
    await persist_turn_to_db(conversation_id, payload.message, content or "")
    return ChatResponse(conversation_id=conversation_id, reply=content or "")
