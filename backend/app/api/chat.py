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


settings = get_settings()
router = APIRouter()

llm_client: LLMClient | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    conversation_id: UUID
    reply: str


async def load_history_from_db(conversation_id: UUID) -> list[dict[str, str]]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        rows = result.all()
        return [{"role": role, "content": content} for role, content in rows]


async def _ensure_conversation(*, session, conversation_id: UUID) -> None:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        session.add(Conversation(id=conversation_id))
        await session.commit()


async def persist_user_message_to_db(conversation_id: UUID, content: str) -> None:
    async with SessionLocal() as session:
        await _ensure_conversation(session=session, conversation_id=conversation_id)
        session.add(
            Message(conversation_id=conversation_id, role="user", content=content)
        )
        await session.commit()


async def persist_assistant_message_to_db(conversation_id: UUID, content: str) -> None:
    async with SessionLocal() as session:
        await _ensure_conversation(session=session, conversation_id=conversation_id)
        session.add(
            Message(conversation_id=conversation_id, role="assistant", content=content)
        )
        await session.commit()


async def persist_turn_to_db(conversation_id: UUID, user: str, assistant: str) -> None:
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
    global llm_client

    if llm_client is None:
        try:
            llm_client = LLMClient.from_settings(settings)
        except ValueError as exc:
            raise ConfigurationError(message=str(exc)) from exc

    conversation_id = payload.conversation_id or uuid4()

    if payload.conversation_id is not None:
        messages = await load_history_from_db(conversation_id)
    else:
        messages = [{"role": m.role, "content": m.content} for m in payload.history]
    messages.append({"role": "user", "content": payload.message})

    if stream:
        await persist_user_message_to_db(conversation_id, payload.message)
        openai_stream = await llm_client.chat(
            messages=messages,
            temperature=0.0,
            stream=True,
        )
        assistant_parts: list[str] = []

        async def _generator() -> AsyncGenerator[str, None]:
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
