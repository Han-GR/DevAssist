from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any

import structlog

from app.core.errors import AppError


DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_S = 5
DEFAULT_MEMORY_LIMIT = "256m"
DEFAULT_TMPFS_SIZE = "64m"
DEFAULT_PIDS_LIMIT = 128
DEFAULT_CPUS = "1"
MAX_OUTPUT_CHARS = 20000


async def execute_python(
    *,
    code: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    image: str = DEFAULT_SANDBOX_IMAGE,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
) -> dict[str, Any]:
    """
    在 Docker 沙箱中执行 Python 代码。

    Args:
        code (str): 待执行的 Python 代码字符串。
        timeout_s (int): 超时时间（秒），默认 5。
        image (str): Docker 镜像名，默认 python:3.12-slim。
        memory_limit (str): Docker 内存限制，默认 256m。

    Returns:
        dict[str, Any]:
            - stdout (str): 标准输出
            - stderr (str): 标准错误
            - exit_code (int): 进程退出码
            - duration_ms (int): 执行耗时（毫秒）

    Raises:
        AppError: 当 docker 不可用、超时或输入不合法时抛出。
        Exception: 其他底层异常原样抛出，由上层统一处理。

    Notes/Examples:
        这是一个“可跑通的最小沙箱”实现：
        - docker run --network none 隔离网络
        - --read-only + --tmpfs /tmp 降低写入面
        - 使用 bind mount 只读挂载代码文件
    """
    if timeout_s <= 0:
        raise AppError(
            code="tool_input_invalid",
            message="timeout_s must be a positive integer.",
            status_code=400,
            details={"timeout_s": timeout_s},
        )
    if not image.strip():
        raise AppError(
            code="tool_input_invalid",
            message="image is required.",
            status_code=400,
            details={"image": image},
        )

    logger = structlog.get_logger()
    start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "main.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            memory_limit,
            "--pids-limit",
            str(DEFAULT_PIDS_LIMIT),
            "--cpus",
            DEFAULT_CPUS,
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,size={DEFAULT_TMPFS_SIZE}",
            "-v",
            f"{tmpdir}:/work:ro",
            "-w",
            "/work",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            image,
            "python",
            "main.py",
        ]

        logger.info(
            "sandbox_start",
            image=image,
            timeout_s=timeout_s,
            memory_limit=memory_limit,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AppError(
                code="docker_not_available",
                message="docker command not found.",
                status_code=500,
                details={"error": str(exc)},
            ) from exc

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise AppError(
                code="sandbox_timeout",
                message="Sandbox execution timed out.",
                status_code=408,
                details={"timeout_s": timeout_s},
            ) from exc

        duration_ms = int((time.perf_counter() - start) * 1000)
        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")

        if len(stdout) > MAX_OUTPUT_CHARS:
            stdout = stdout[:MAX_OUTPUT_CHARS]
        if len(stderr) > MAX_OUTPUT_CHARS:
            stderr = stderr[:MAX_OUTPUT_CHARS]

        exit_code = int(proc.returncode or 0)
        logger.info(
            "sandbox_done",
            exit_code=exit_code,
            duration_ms=duration_ms,
            stdout_chars=len(stdout),
            stderr_chars=len(stderr),
        )

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
        }

