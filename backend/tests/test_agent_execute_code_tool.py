from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.builtin_tools import create_execute_code_tool
from app.core.errors import AppError


def test_execute_code_tool_invalid_timeout_raises_app_error() -> None:
    tool = create_execute_code_tool()
    with pytest.raises(AppError) as exc:
        asyncio.run(tool.call({"code": "print(1)", "timeout_s": 0}))
    assert exc.value.code == "tool_input_invalid"


def test_execute_code_tool_calls_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_execute_python(*, code: str, timeout_s: int) -> dict[str, object]:
        assert code.strip() == "print('ok')"
        assert timeout_s == 3
        return {"stdout": "ok\n", "stderr": "", "exit_code": 0, "duration_ms": 12}

    import app.agent.builtin_tools as module

    monkeypatch.setattr(module, "execute_python", fake_execute_python)

    tool = create_execute_code_tool()
    out = asyncio.run(tool.call({"code": "print('ok')", "timeout_s": 3}))
    assert out["stdout"] == "ok\n"
    assert out["exit_code"] == 0

