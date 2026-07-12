from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any
from uuid import uuid4

import docker
from docker.errors import DockerException
import structlog

from app.core.errors import AppError
from app.core.config import get_settings


DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_S = 5
DEFAULT_MEMORY_LIMIT = "256m"
DEFAULT_TMPFS_SIZE = "64m"
DEFAULT_PIDS_LIMIT = 128
MAX_OUTPUT_CHARS = 20000


async def execute_python(
    *,
    code: str,
    timeout_s: int | None = None,
    image: str | None = None,
    memory_limit: str | None = None,
) -> dict[str, Any]:
    """
    在 Docker 沙箱中执行 Python 代码。

    Args:
        code (str): 待执行的 Python 代码字符串。
        timeout_s (int | None): 超时时间（秒）；不传则从 Settings 读取，默认 5。
        image (str | None): Docker 镜像名；不传则从 Settings 读取，默认 python:3.12-slim。
        memory_limit (str | None): Docker 内存限制；不传则从 Settings 读取，默认 256m。

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
        当前使用 Docker SDK（docker-py）以便后续更精细地控制资源隔离与日志采集。
    """
    settings = get_settings()
    timeout_s = settings.sandbox_timeout if timeout_s is None else timeout_s
    image = settings.sandbox_image if image is None else image
    memory_limit = settings.sandbox_memory_limit if memory_limit is None else memory_limit

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

        logger.info(
            "sandbox_start",
            image=image,
            timeout_s=timeout_s,
            memory_limit=memory_limit,
        )

        try:
            client = docker.from_env()
        except DockerException as exc:
            raise AppError(
                code="docker_not_available",
                message="Docker is not available.",
                status_code=500,
                details={"error": str(exc)},
            ) from exc
        container = None
        try:
            container = await asyncio.to_thread(
                client.containers.create,
                image=image,
                command=["python", "main.py"],
                name=f"devassist-sandbox-{uuid4().hex[:12]}",
                detach=True,
                network_disabled=True,
                mem_limit=memory_limit,
                pids_limit=DEFAULT_PIDS_LIMIT,
                read_only=True,
                tmpfs={"/tmp": f"rw,size={DEFAULT_TMPFS_SIZE}"},
                volumes={tmpdir: {"bind": "/work", "mode": "ro"}},
                working_dir="/work",
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            await asyncio.to_thread(container.start)
            wait_result = await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=timeout_s,
            )
            exit_code = int((wait_result or {}).get("StatusCode", 0))

            stdout_b = await asyncio.to_thread(container.logs, stdout=True, stderr=False)
            stderr_b = await asyncio.to_thread(container.logs, stdout=False, stderr=True)
        except asyncio.TimeoutError as exc:
            if container is not None:
                try:
                    await asyncio.to_thread(container.kill)
                finally:
                    await asyncio.to_thread(container.remove, force=True)
            raise AppError(
                code="sandbox_timeout",
                message="Sandbox execution timed out.",
                status_code=408,
                details={"timeout_s": timeout_s},
            ) from exc
        except DockerException as exc:
            raise AppError(
                code="sandbox_error",
                message="Sandbox execution failed.",
                status_code=500,
                details={"error": str(exc)},
            ) from exc
        finally:
            if container is not None:
                try:
                    await asyncio.to_thread(container.remove, force=True)
                except Exception:
                    pass

        duration_ms = int((time.perf_counter() - start) * 1000)
        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")

        if len(stdout) > MAX_OUTPUT_CHARS:
            stdout = stdout[:MAX_OUTPUT_CHARS]
        if len(stderr) > MAX_OUTPUT_CHARS:
            stderr = stderr[:MAX_OUTPUT_CHARS]

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
