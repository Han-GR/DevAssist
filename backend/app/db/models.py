"""
数据库模型定义（SQLAlchemy ORM）。

目前只做聊天持久化的最小集合：
- conversations：会话元信息
- messages：会话中的消息（user/assistant/system）
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TimestampMixin:
    """
    通用时间戳字段。

    统一用数据库时间（server_default=now）是为了避免应用服务器时钟偏差带来的混乱，
    也方便后面做排序和回放。
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Conversation(TimestampMixin, Base):
    """
    会话表：一段连续对话的“壳”。

    目前先保留 user_id/title 这些字段，后续接入用户系统或做会话列表时会用到。
    """

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    title: Mapped[str] = mapped_column(String(200), default="", nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class Message(TimestampMixin, Base):
    """
    消息表：会话里的每一条消息。

    citations 预留给后续 RAG，用 JSONB 存来源信息，结构可以灵活演进。
    """

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Document(TimestampMixin, Base):
    """
    文档表：记录一次 ingestion 的元信息。

    当前阶段主要用途是“可追踪”：知道哪些文件被写入了知识库，以及大概切了多少 chunk。
    后续如果要做删除/重建索引/版本化，这张表会成为核心索引入口。
    """

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    source: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class AgentTrace(TimestampMixin, Base):
    """
    Agent Trace 表：记录一次 Agent 运行的完整步骤与最终结果。

    设计要点：
    - steps 用 JSONB 存储，便于后续在前端做“逐步回放”
    - conversation_id 可选：Agent 既可以独立调用，也可以挂在某个 chat 会话下
    """

    __tablename__ = "agent_traces"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, unique=True, index=True)
    conversation_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, default="react")
    steps: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False, default=list)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
