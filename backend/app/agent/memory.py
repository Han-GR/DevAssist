from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Literal
from uuid import UUID


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class MemoryMessage:
    """
    短期记忆中的一条消息。

    Args:
        role (Role): 角色（system/user/assistant）。
        content (str): 消息内容。

    Returns:
        MemoryMessage: 消息对象。

    Raises:
        None
    """

    role: Role
    content: str

    def to_openai_dict(self) -> dict[str, str]:
        """
        导出为 OpenAI-style messages 的一项。

        Args:
            None

        Returns:
            dict[str, str]: {"role": "...", "content": "..."}。

        Raises:
            None
        """
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """
    短期记忆：按 conversation_id 保存最近 N 个“turn”。

    Args:
        max_turns (int): 最大 turn 数（turn = user + assistant 两条消息）。
        max_conversations (int): 最大会话数（超出时按 LRU 淘汰）。

    Returns:
        ShortTermMemory: 记忆对象。

    Raises:
        ValueError: 参数非法时抛出。
    """

    def __init__(self, *, max_turns: int = 20, max_conversations: int = 200) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if max_conversations <= 0:
            raise ValueError("max_conversations must be a positive integer")

        self._max_turns = max_turns
        self._max_messages = max_turns * 2
        self._max_conversations = max_conversations
        self._store: OrderedDict[UUID, deque[MemoryMessage]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def append(self, *, conversation_id: UUID, role: Role, content: str) -> None:
        """
        追加一条消息到指定会话的短期记忆中。

        Args:
            conversation_id (UUID): 会话 ID。
            role (Role): 角色（system/user/assistant）。
            content (str): 消息内容。

        Returns:
            None

        Raises:
            None
        """
        if not content:
            return

        async with self._lock:
            bucket = self._store.get(conversation_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_messages)
                self._store[conversation_id] = bucket
            else:
                self._store.move_to_end(conversation_id, last=True)

            bucket.append(MemoryMessage(role=role, content=content))
            await self._evict_if_needed_locked()

    async def add_turn(self, *, conversation_id: UUID, user: str, assistant: str) -> None:
        """
        追加一个 turn（user + assistant）。

        Args:
            conversation_id (UUID): 会话 ID。
            user (str): 用户输入。
            assistant (str): 助手回复。

        Returns:
            None

        Raises:
            None
        """
        await self.append(conversation_id=conversation_id, role="user", content=user)
        await self.append(conversation_id=conversation_id, role="assistant", content=assistant)

    async def get_history(self, *, conversation_id: UUID) -> list[dict[str, str]]:
        """
        获取指定会话的历史消息（OpenAI-style messages）。

        Args:
            conversation_id (UUID): 会话 ID。

        Returns:
            list[dict[str, str]]: 形如 [{"role": "...", "content": "..."}, ...]。

        Raises:
            None
        """
        async with self._lock:
            bucket = self._store.get(conversation_id)
            if bucket is None:
                return []
            self._store.move_to_end(conversation_id, last=True)
            return [m.to_openai_dict() for m in list(bucket)]

    async def clear(self, *, conversation_id: UUID | None = None) -> None:
        """
        清空短期记忆（测试/调试用）。

        Args:
            conversation_id (UUID | None): 若传入则只清空该会话；否则清空全部。

        Returns:
            None

        Raises:
            None
        """
        async with self._lock:
            if conversation_id is None:
                self._store.clear()
            else:
                self._store.pop(conversation_id, None)

    async def _evict_if_needed_locked(self) -> None:
        if len(self._store) <= self._max_conversations:
            return
        while len(self._store) > self._max_conversations:
            self._store.popitem(last=False)


short_term_memory = ShortTermMemory()

