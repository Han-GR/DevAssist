"""
流式输出相关工具。

这里主要做两件事：
1) 把业务数据包装成 SSE 协议格式；
2) 把 OpenAI-style 的 streaming response 转成前端更好消费的 SSE 事件流。
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

import structlog


def sse_event(*, data: Any, event: str | None = None) -> str:
    """
    把一条事件格式化为 SSE 文本。

    Args:
        data (Any): 要发送给前端的数据，会被序列化为 JSON。
        event (str | None): 可选的 SSE event 名称；不传则只发送 data 字段。

    Returns:
        str: 符合 SSE 协议的文本片段（以空行结尾）。

    Raises:
        TypeError: 当 data 无法被 JSON 序列化时可能抛出。

    Notes:
        我们统一把 data 序列化成 JSON（而不是直接拼字符串），是为了让前端处理更一致：
        - meta/delta/done/error 都是同一种解析方式
        - 也便于后续在不改协议的前提下扩字段
    """
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
    """
    将 OpenAI SDK 的 chat.completions streaming 输出转成 SSE。

    Args:
        stream (AsyncIterable[Any]): OpenAI SDK 返回的流式对象（可 async for）。
        conversation_id (str | None): 会话 ID；传入时会先发送 meta 包，方便前端保存。

    Yields:
        str: SSE 文本片段（meta/delta/done/error）。

    Raises:
        Exception: 内部会捕获异常并输出 error 事件，但调用方的 stream 也可能在迭代时抛出异常。

    Notes:
        约定的事件类型：
        - meta：会话信息（可选）
        - delta：增量文本
        - done：流结束
        - error：发生异常时的兜底信号
    """
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
