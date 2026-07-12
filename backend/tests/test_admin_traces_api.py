from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.admin_traces as admin_module


@dataclass
class _FakeTrace:
    run_id: UUID
    conversation_id: UUID | None
    agent_type: str
    steps: list[dict[str, Any]]
    result: str | None
    error: str | None
    created_at: str


def test_admin_agent_traces_returns_list() -> None:
    client = TestClient(main_module.app)
    run_id = uuid4()

    async def _fake_list_agent_traces_from_db(*, limit: int = 50):
        _ = limit
        return [
            _FakeTrace(
                run_id=run_id,
                conversation_id=None,
                agent_type="react",
                steps=[{"step_index": 0, "thought": "x"}],
                result="ok",
                error=None,
                created_at="2026-07-12T00:00:00+00:00",
            )
        ]

    admin_module.list_agent_traces_from_db = _fake_list_agent_traces_from_db  # type: ignore[assignment]

    resp = client.get("/admin/agent-traces?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert body[0]["run_id"] == str(run_id)
    assert body[0]["steps"][0]["step_index"] == 0
