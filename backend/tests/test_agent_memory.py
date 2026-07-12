from __future__ import annotations

import asyncio
from uuid import uuid4

from app.agent.memory import ShortTermMemory


def test_short_term_memory_keeps_last_turns() -> None:
    mem = ShortTermMemory(max_turns=2, max_conversations=10)
    cid = uuid4()

    asyncio.run(mem.add_turn(conversation_id=cid, user="u1", assistant="a1"))
    asyncio.run(mem.add_turn(conversation_id=cid, user="u2", assistant="a2"))
    asyncio.run(mem.add_turn(conversation_id=cid, user="u3", assistant="a3"))

    history = asyncio.run(mem.get_history(conversation_id=cid))
    assert history == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_short_term_memory_lru_eviction() -> None:
    mem = ShortTermMemory(max_turns=1, max_conversations=2)
    cid1 = uuid4()
    cid2 = uuid4()
    cid3 = uuid4()

    asyncio.run(mem.add_turn(conversation_id=cid1, user="u1", assistant="a1"))
    asyncio.run(mem.add_turn(conversation_id=cid2, user="u2", assistant="a2"))
    asyncio.run(mem.add_turn(conversation_id=cid3, user="u3", assistant="a3"))

    assert asyncio.run(mem.get_history(conversation_id=cid1)) == []
    assert asyncio.run(mem.get_history(conversation_id=cid2))
    assert asyncio.run(mem.get_history(conversation_id=cid3))
