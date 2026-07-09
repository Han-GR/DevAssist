from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

import structlog


def sse_event(*, data: Any, event: str | None = None) -> str:
    # SSE 协议其实就两行：event/data，加个空行结束一条消息
    # 统一把 data 包成 JSON，前端解析也更省事
    payload = json.dumps(data, ensure_ascii=False)

    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


async def openai_chat_stream_to_sse(
    stream: AsyncIterable[Any],
    *,
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    logger = structlog.get_logger()

    try:
        if conversation_id:
            # 先把会话信息丢给前端，后面断线重连也好对上
            yield sse_event(data={"type": "meta", "conversation_id": conversation_id})

        async for chunk in stream:
            # OpenAI-style 的流式返回是增量 delta，一次可能只吐几个字
            delta = None
            try:
                delta = chunk.choices[0].delta
            except Exception:
                delta = None

            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield sse_event(data={"type": "delta", "content": content})

        # 给前端一个明确的收尾信号，别让它一直转圈
        yield sse_event(data={"type": "done"}, event="done")
    except Exception as exc:
        logger.exception("sse_stream_error", error=str(exc))
        # 出错也按 SSE 往外吐一条，前端至少能优雅停掉
        yield sse_event(data={"type": "error", "message": "stream_error"}, event="error")
