from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID


@dataclass(frozen=True)
class SFTSample:
    instruction: str
    input: str
    output: str
    meta: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """
        转为可写入 JSONL 的 dict。

        Args:
            无。

        Returns:
            包含 instruction/input/output 的对象；若 meta 存在则带上 meta。

        Raises:
            无。

        Notes:
            - 该结构与 `finetune/README.md` 的 SFT Schema 保持一致。
        """
        payload: dict[str, Any] = {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }
        if self.meta:
            payload["meta"] = self.meta
        return payload


def build_sft_samples_from_messages(
    *,
    messages: Sequence[Mapping[str, Any]],
    instruction: str,
    conversation_id: str | None = None,
    include_meta: bool = True,
) -> list[SFTSample]:
    """
    从一段消息序列构造 SFT 样本（user -> assistant 配对）。

    Args:
        messages: 按时间顺序排列的消息列表。每条至少包含 role/content，可选 id/created_at 等字段。
        instruction: SFT 的系统指令（建议统一且短）。
        conversation_id: 可选，会写入 meta，便于追踪来源。
        include_meta: 是否输出 meta 字段。

    Returns:
        SFTSample 列表，每个样本由一对 user/assistant 消息组成。

    Raises:
        ValueError: instruction 为空，或 messages 结构不合法。

    Notes:
        - 仅处理 role 为 user/assistant 的消息。
        - 如果 user 后面没有紧跟 assistant，会跳过该 user。
    """

    inst = (instruction or "").strip()
    if not inst:
        raise ValueError("instruction must not be empty")

    items: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    prev_user: Mapping[str, Any] | None = None

    for m in messages:
        role = str(m.get("role") or "").strip()
        if role == "user":
            prev_user = m
            continue
        if role == "assistant" and prev_user is not None:
            items.append((prev_user, m))
            prev_user = None

    samples: list[SFTSample] = []
    for user_msg, assistant_msg in items:
        user_text = str(user_msg.get("content") or "").strip()
        assistant_text = str(assistant_msg.get("content") or "").strip()
        if not user_text or not assistant_text:
            continue

        meta: dict[str, Any] | None = None
        if include_meta:
            meta = {
                "source": str(user_msg.get("source") or "chat_export"),
                "conversation_id": conversation_id,
                "message_ids": [
                    str(user_msg.get("id") or ""),
                    str(assistant_msg.get("id") or ""),
                ],
                "created_at": str(assistant_msg.get("created_at") or user_msg.get("created_at") or ""),
            }
            meta = {k: v for k, v in meta.items() if v not in (None, "", [])}

        samples.append(SFTSample(instruction=inst, input=user_text, output=assistant_text, meta=meta))

    return samples


async def export_sft_jsonl_from_db(
    *,
    output_path: Path,
    instruction: str,
    conversation_limit: int | None = None,
    include_meta: bool = True,
    since: datetime | None = None,
    until: datetime | None = None,
) -> int:
    """
    从数据库导出 SFT JSONL 数据集（conversation -> messages -> samples）。

    Args:
        output_path: 目标 JSONL 文件路径。
        instruction: SFT 的系统指令（建议统一且短）。
        conversation_limit: 限制导出的会话数量（按 created_at 倒序）。
        include_meta: 是否在样本里附加 meta（conversation_id/message_ids 等）。
        since: 仅导出 created_at >= since 的会话。
        until: 仅导出 created_at <= until 的会话。

    Returns:
        写入的样本条数。

    Raises:
        ValueError: instruction 为空。
        OSError: 输出目录不可写等文件系统错误。

    Notes:
        - 为了便于追溯与调试，默认 include_meta=true。
    """

    inst = (instruction or "").strip()
    if not inst:
        raise ValueError("instruction must not be empty")

    from sqlalchemy import select

    from app.db.models import Conversation, Message
    from app.db.session import SessionLocal

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with SessionLocal() as session:
        stmt = select(Conversation.id, Conversation.created_at).order_by(Conversation.created_at.desc())
        if since is not None:
            stmt = stmt.where(Conversation.created_at >= since)
        if until is not None:
            stmt = stmt.where(Conversation.created_at <= until)
        if conversation_limit is not None:
            stmt = stmt.limit(conversation_limit)

        rows = (await session.execute(stmt)).all()
        conversation_ids: list[UUID] = [r[0] for r in rows]

        total = 0
        with output_path.open("w", encoding="utf-8") as f:
            for cid in conversation_ids:
                m_stmt = (
                    select(Message.id, Message.role, Message.content, Message.created_at)
                    .where(Message.conversation_id == cid)
                    .order_by(Message.created_at.asc())
                )
                m_rows = (await session.execute(m_stmt)).all()
                raw_messages: list[dict[str, Any]] = [
                    {"id": mid, "role": role, "content": content, "created_at": created_at}
                    for (mid, role, content, created_at) in m_rows
                ]

                samples = build_sft_samples_from_messages(
                    messages=raw_messages,
                    instruction=inst,
                    conversation_id=str(cid),
                    include_meta=include_meta,
                )
                for s in samples:
                    f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
                total += len(samples)

    return total
