from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

import structlog

from app.core.config import get_settings
from app.core.llm import LLMClient
from app.rag.chroma import ChromaCollectionManager
from app.rag.embedder import Embedder


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

    async def append(self, *, conversation_id: UUID, role: Role, content: str) -> list[MemoryMessage]:
        """
        追加一条消息到指定会话的短期记忆中。

        Args:
            conversation_id (UUID): 会话 ID。
            role (Role): 角色（system/user/assistant）。
            content (str): 消息内容。

        Returns:
            list[MemoryMessage]: 本次写入导致被挤出的消息（可能为空）。

        Raises:
            None
        """
        if not content:
            return []

        async with self._lock:
            bucket = self._store.get(conversation_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_messages)
                self._store[conversation_id] = bucket
            else:
                self._store.move_to_end(conversation_id, last=True)

            evicted: list[MemoryMessage] = []
            if bucket.maxlen is not None and len(bucket) == bucket.maxlen:
                evicted.append(bucket[0])
            bucket.append(MemoryMessage(role=role, content=content))
            await self._evict_if_needed_locked()
            return evicted

    async def add_turn(self, *, conversation_id: UUID, user: str, assistant: str) -> list[MemoryMessage]:
        """
        追加一个 turn（user + assistant）。

        Args:
            conversation_id (UUID): 会话 ID。
            user (str): 用户输入。
            assistant (str): 助手回复。

        Returns:
            list[MemoryMessage]: 本次写入导致被挤出的消息（可能为空）。

        Raises:
            None
        """
        evicted: list[MemoryMessage] = []
        evicted.extend(await self.append(conversation_id=conversation_id, role="user", content=user))
        evicted.extend(await self.append(conversation_id=conversation_id, role="assistant", content=assistant))
        return evicted

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


class LongTermMemory:
    """
    长时记忆：把“历史摘要”写入 Chroma，后续可按 query 检索相关记忆。

    Args:
        embedder (Embedder): embedding 生成器。
        chroma (ChromaCollectionManager): Chroma collection 管理器。
        collection_name (str): collection 名称。

    Returns:
        LongTermMemory: 长时记忆对象。

    Raises:
        ValueError: collection_name 为空时抛出。
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        chroma: ChromaCollectionManager,
        collection_name: str = "agent_memory",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name is required")
        self._embedder = embedder
        self._chroma = chroma
        self._collection_name = collection_name
        self._logger = structlog.get_logger()

    async def add_summary(self, *, conversation_id: UUID, summary: str) -> None:
        """
        写入一条长时记忆摘要。

        Args:
            conversation_id (UUID): 会话 ID。
            summary (str): 摘要文本。

        Returns:
            None

        Raises:
            Exception: embedding 或 Chroma 写入失败时抛出。
        """
        if not summary.strip():
            return

        vectors = await self._embedder.embed_texts([summary])
        vector = vectors[0] if vectors else []
        doc_id = uuid4().hex
        created_at_ms = int(time.time() * 1000)

        collection = await asyncio.to_thread(
            self._chroma.get_or_create_collection,
            name=self._collection_name,
        )
        await asyncio.to_thread(
            collection.add,
            ids=[doc_id],
            documents=[summary],
            embeddings=[vector],
            metadatas=[
                {
                    "conversation_id": str(conversation_id),
                    "kind": "summary",
                    "created_at_ms": created_at_ms,
                }
            ],
        )
        self._logger.info(
            "long_term_memory_added",
            conversation_id=str(conversation_id),
            doc_id=doc_id,
        )

    async def search(self, *, conversation_id: UUID, query: str, top_k: int = 3) -> list[str]:
        """
        按 query 检索相关长时记忆摘要。

        Args:
            conversation_id (UUID): 会话 ID。
            query (str): 当前用户问题。
            top_k (int): 返回条数。

        Returns:
            list[str]: 摘要列表（按相关性排序）。

        Raises:
            Exception: embedding 或 Chroma 查询失败时抛出。
        """
        if not query.strip():
            return []
        if top_k <= 0:
            return []

        vectors = await self._embedder.embed_texts([query])
        vector = vectors[0] if vectors else []
        collection = await asyncio.to_thread(
            self._chroma.get_or_create_collection,
            name=self._collection_name,
        )
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[vector],
            n_results=top_k,
            where={"conversation_id": str(conversation_id)},
        )
        docs = result.get("documents") if isinstance(result, dict) else None
        if not docs or not isinstance(docs, list) or not docs[0]:
            return []
        first = docs[0]
        return [str(x) for x in first if x]


def _try_build_long_term_memory() -> LongTermMemory | None:
    settings = get_settings()
    if not settings.embedding_api_key or not settings.embedding_model:
        return None

    try:
        embedder = Embedder.from_settings(settings)
        chroma = ChromaCollectionManager.from_settings(settings)
        return LongTermMemory(embedder=embedder, chroma=chroma)
    except Exception:
        return None


class MemoryManager:
    """
    记忆管理器：短期 + 长期记忆协作。

    Args:
        short_term (ShortTermMemory): 短期记忆实现。
        long_term (LongTermMemory | None): 长期记忆实现；为 None 时表示禁用长时记忆。
        summarize_min_messages (int): 触发一次摘要所需的最小消息数。

    Returns:
        MemoryManager: 管理器实例。

    Raises:
        ValueError: summarize_min_messages 非法时抛出。
    """

    def __init__(
        self,
        *,
        short_term: ShortTermMemory,
        long_term: LongTermMemory | None,
        summarize_min_messages: int = 12,
    ) -> None:
        if summarize_min_messages <= 0:
            raise ValueError("summarize_min_messages must be a positive integer")
        self._short = short_term
        self._long = long_term
        self._summarize_min_messages = summarize_min_messages
        self._archive_buffers: dict[UUID, list[MemoryMessage]] = {}
        self._logger = structlog.get_logger()

    async def build_history(self, *, conversation_id: UUID, query: str) -> list[dict[str, str]]:
        """
        构建本次推理的历史上下文（长时记忆 + 短期记忆）。

        Args:
            conversation_id (UUID): 会话 ID。
            query (str): 本轮用户问题（用于检索长时记忆）。

        Returns:
            list[dict[str, str]]: OpenAI-style messages，可直接拼进 LLM messages。
        """
        history = await self._short.get_history(conversation_id=conversation_id)
        if self._long is None:
            return history

        try:
            memories = await self._long.search(conversation_id=conversation_id, query=query, top_k=3)
        except Exception as exc:
            self._logger.error("long_term_memory_search_failed", conversation_id=str(conversation_id), error=str(exc))
            return history

        if not memories:
            return history
        text = "\n".join([f"- {m.strip()}" for m in memories if m and m.strip()])
        if not text:
            return history
        return [{"role": "system", "content": f"Long-term memory:\n{text}"}] + history

    async def add_turn(self, *, conversation_id: UUID, user: str, assistant: str, llm: LLMClient) -> None:
        """
        写入一个 turn，并在短期记忆溢出时把被挤出的内容摘要进长时记忆。

        Args:
            conversation_id (UUID): 会话 ID。
            user (str): 用户输入。
            assistant (str): 助手回复。
            llm (LLMClient): 用于摘要的 LLM 客户端。

        Returns:
            None
        """
        evicted = await self._short.add_turn(conversation_id=conversation_id, user=user, assistant=assistant)
        if not evicted:
            return
        if self._long is None:
            return

        buf = self._archive_buffers.get(conversation_id)
        if buf is None:
            buf = []
            self._archive_buffers[conversation_id] = buf
        buf.extend(evicted)
        if len(buf) < self._summarize_min_messages:
            return

        summary = await self._summarize_messages(llm=llm, messages=buf)
        buf.clear()
        if not summary.strip():
            return
        try:
            await self._long.add_summary(conversation_id=conversation_id, summary=summary)
        except Exception as exc:
            self._logger.error("long_term_memory_add_failed", conversation_id=str(conversation_id), error=str(exc))

    async def _summarize_messages(self, *, llm: LLMClient, messages: list[MemoryMessage]) -> str:
        text = "\n".join([f"{m.role}: {m.content}".strip() for m in messages if m.content])
        resp = await llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "\n".join(
                        [
                            "你是 DevAssist 的记忆摘要器。",
                            "请把下面的对话片段总结成“可复用的长期记忆”，要求：",
                            "- 只保留稳定事实/偏好/长期目标，忽略寒暄与临时细节",
                            "- 用中文输出，尽量精炼，最多 8 条要点",
                            "- 不要编造",
                        ]
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            stream=False,
        )
        return str(resp.choices[0].message.content or "").strip()


agent_memory = MemoryManager(short_term=short_term_memory, long_term=_try_build_long_term_memory())
