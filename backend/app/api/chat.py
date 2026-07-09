from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.core.llm import LLMClient
from app.core.streaming import openai_chat_stream_to_sse


settings = get_settings()
router = APIRouter()

llm_client: LLMClient | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


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

    conversation_id = payload.conversation_id or str(uuid4())

    messages: list[dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in payload.history
    ]
    messages.append({"role": "user", "content": payload.message})

    if stream:
        openai_stream = await llm_client.chat(
            messages=messages,
            temperature=0.0,
            stream=True,
        )
        generator = openai_chat_stream_to_sse(
            openai_stream, conversation_id=conversation_id
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    response = await llm_client.chat(
        messages=messages,
        temperature=0.0,
        stream=False,
    )
    content = response.choices[0].message.content if response.choices else ""
    return ChatResponse(conversation_id=conversation_id, reply=content or "")
