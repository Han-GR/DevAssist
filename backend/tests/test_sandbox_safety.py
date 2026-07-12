"""
Day54 — 沙箱安全控制单测。

验证：
1. 危险操作（文件写入、网络请求、系统命令）被正确检测为 blocked。
2. 警告级别操作（eval/exec）被检测为 warning，不 blocked。
3. 安全代码不产生任何 issue。
4. 文件路径白名单：路径在白名单内 → 通过；路径不在白名单 → blocked。
5. execute_code 工具 handler：blocked 代码抛 AppError；warning 代码正常执行（fake sandbox）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.sandbox_safety import (
    SafetyCheckResult,
    check_code_safety,
    format_safety_report,
)
from app.core.errors import AppError


# ---------------------------------------------------------------------------
# 1. 危险操作检测 — blocked
# ---------------------------------------------------------------------------

def test_file_write_is_blocked() -> None:
    code = 'with open("/etc/passwd", "w") as f:\n    f.write("hacked")'
    result = check_code_safety(code=code)
    assert result.is_blocked
    blocked = [i for i in result.issues if i.level == "blocked"]
    assert any("file write" in i.reason for i in blocked)


def test_network_access_is_blocked() -> None:
    code = "import requests\nrequests.get('http://evil.com')"
    result = check_code_safety(code=code)
    assert result.is_blocked
    assert any("network" in i.reason for i in result.issues)


def test_subprocess_is_blocked() -> None:
    code = "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"
    result = check_code_safety(code=code)
    assert result.is_blocked
    assert any("subprocess" in i.reason for i in result.issues)


def test_os_system_is_blocked() -> None:
    code = "import os\nos.system('rm -rf /')"
    result = check_code_safety(code=code)
    assert result.is_blocked
    assert any("system command" in i.reason for i in result.issues)


# ---------------------------------------------------------------------------
# 2. 警告级别操作 — warning，不 blocked
# ---------------------------------------------------------------------------

def test_eval_is_warning_not_blocked() -> None:
    code = "result = eval('1 + 2')"
    result = check_code_safety(code=code)
    assert not result.is_blocked
    assert any(i.level == "warning" and "eval" in i.reason for i in result.issues)


def test_exec_is_warning_not_blocked() -> None:
    code = "exec('x = 1')"
    result = check_code_safety(code=code)
    assert not result.is_blocked
    assert any(i.level == "warning" and "exec" in i.reason for i in result.issues)


# ---------------------------------------------------------------------------
# 3. 安全代码 — 无 issue
# ---------------------------------------------------------------------------

def test_safe_code_has_no_issues() -> None:
    code = "x = [i**2 for i in range(10)]\nprint(sum(x))"
    result = check_code_safety(code=code)
    assert not result.is_blocked
    assert result.issues == []


# ---------------------------------------------------------------------------
# 4. 文件路径白名单
# ---------------------------------------------------------------------------

def test_path_in_allowlist_passes() -> None:
    code = 'with open("/allowed/data.txt", "r") as f:\n    data = f.read()'
    # 只读操作不触发 blocked，路径在白名单内也不触发
    result = check_code_safety(code=code, allowed_paths=["/allowed"])
    # open("r") 不在危险模式里，路径在白名单内 → 无 blocked
    assert not result.is_blocked


def test_path_outside_allowlist_is_blocked() -> None:
    code = 'data = open("/etc/shadow", "r").read()'
    result = check_code_safety(code=code, allowed_paths=["/allowed"])
    assert result.is_blocked
    assert any("not in allowlist" in i.reason for i in result.issues)


def test_empty_allowlist_skips_path_check() -> None:
    code = 'data = open("/etc/shadow", "r").read()'
    # 空白名单 → 不做路径检查（open("r") 本身不是危险模式）
    result = check_code_safety(code=code, allowed_paths=[])
    assert not result.is_blocked


# ---------------------------------------------------------------------------
# 5. format_safety_report
# ---------------------------------------------------------------------------

def test_format_safety_report_no_issues() -> None:
    result = SafetyCheckResult(is_blocked=False, issues=[])
    report = format_safety_report(result)
    assert "No safety issues" in report


def test_format_safety_report_with_issues() -> None:
    code = "import subprocess\nsubprocess.run(['ls'])"
    result = check_code_safety(code=code)
    report = format_safety_report(result)
    assert "BLOCKED" in report
    assert "subprocess" in report


# ---------------------------------------------------------------------------
# 6. execute_code 工具 handler：blocked 代码抛 AppError
# ---------------------------------------------------------------------------

def test_execute_code_tool_blocks_dangerous_code() -> None:
    """execute_code handler 对 blocked 代码应抛出 AppError(sandbox_code_blocked)。"""
    from app.agent.builtin_tools import create_execute_code_tool  # noqa: PLC0415

    tool = create_execute_code_tool()

    dangerous_code = "import subprocess\nsubprocess.run(['ls'])"

    async def _run() -> None:
        await tool.handler(code=dangerous_code, timeout_s=5)  # type: ignore[misc]

    try:
        asyncio.run(_run())
        assert False, "Should have raised AppError"
    except AppError as exc:
        assert exc.code == "sandbox_code_blocked"


def test_execute_code_tool_allows_safe_code() -> None:
    """execute_code handler 对安全代码应调用 execute_python（fake sandbox）。"""
    from app.agent.builtin_tools import create_execute_code_tool  # noqa: PLC0415
    import app.agent.builtin_tools as bt_module  # noqa: PLC0415

    tool = create_execute_code_tool()
    safe_code = "print('hello')"
    called_with: dict[str, Any] = {}

    async def _fake_execute_python(**kwargs: Any) -> dict[str, Any]:
        called_with.update(kwargs)
        return {"stdout": "hello\n", "stderr": "", "exit_code": 0, "duration_ms": 10}

    with patch.object(bt_module, "execute_python", _fake_execute_python):
        result = asyncio.run(tool.handler(code=safe_code, timeout_s=5))  # type: ignore[misc]

    assert result["exit_code"] == 0
    assert called_with.get("code") == safe_code
