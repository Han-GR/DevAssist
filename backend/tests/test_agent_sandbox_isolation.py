from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.errors import AppError


def test_sandbox_execute_python_uses_isolation_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.sandbox as sandbox

    class FakeContainer:
        def __init__(self) -> None:
            self.started = False
            self.removed = False
            self.killed = False
            self.logs_calls: list[dict[str, bool]] = []

        def start(self) -> None:
            self.started = True

        def wait(self) -> dict[str, int]:
            return {"StatusCode": 0}

        def logs(self, *, stdout: bool, stderr: bool) -> bytes:
            self.logs_calls.append({"stdout": stdout, "stderr": stderr})
            if stdout and not stderr:
                return b"hello\n"
            if stderr and not stdout:
                return b""
            return b""

        def remove(self, *, force: bool) -> None:
            self.removed = True

        def kill(self) -> None:
            self.killed = True

    class FakeContainers:
        def __init__(self) -> None:
            self.last_create_kwargs: dict[str, object] | None = None
            self.container = FakeContainer()

        def create(self, **kwargs: object) -> FakeContainer:
            self.last_create_kwargs = kwargs
            return self.container

    class FakeClient:
        def __init__(self) -> None:
            self.containers = FakeContainers()

    fake_client = FakeClient()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(sandbox.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(sandbox.asyncio, "to_thread", fake_to_thread)

    out = asyncio.run(
        sandbox.execute_python(
            code="print('hi')",
            timeout_s=5,
            image="python:3.12-slim",
            memory_limit="256m",
        )
    )

    assert out["exit_code"] == 0
    assert out["stdout"] == "hello\n"
    assert out["stderr"] == ""

    create_kwargs = fake_client.containers.last_create_kwargs
    assert create_kwargs is not None
    assert create_kwargs["network_disabled"] is True
    assert create_kwargs["read_only"] is True
    assert create_kwargs["working_dir"] == "/work"
    assert create_kwargs["mem_limit"] == "256m"
    assert create_kwargs["pids_limit"] == sandbox.DEFAULT_PIDS_LIMIT
    assert create_kwargs["tmpfs"] == {"/tmp": f"rw,size={sandbox.DEFAULT_TMPFS_SIZE}"}

    volumes = create_kwargs["volumes"]
    assert isinstance(volumes, dict)
    assert list(volumes.values())[0] == {"bind": "/work", "mode": "ro"}

    env = create_kwargs["environment"]
    assert env == {"PYTHONDONTWRITEBYTECODE": "1"}

    calls = fake_client.containers.container.logs_calls
    assert {"stdout": True, "stderr": False} in calls
    assert {"stdout": False, "stderr": True} in calls
    assert fake_client.containers.container.removed is True


def test_sandbox_execute_python_timeout_kills_container(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.sandbox as sandbox

    class FakeContainer:
        def __init__(self) -> None:
            self.killed = False
            self.removed = False

        def start(self) -> None:
            return None

        def wait(self) -> dict[str, int]:
            return {"StatusCode": 0}

        def logs(self, *, stdout: bool, stderr: bool) -> bytes:
            return b""

        def kill(self) -> None:
            self.killed = True

        def remove(self, *, force: bool) -> None:
            self.removed = True

    class FakeContainers:
        def __init__(self) -> None:
            self.container = FakeContainer()

        def create(self, **kwargs: object) -> FakeContainer:
            return self.container

    class FakeClient:
        def __init__(self) -> None:
            self.containers = FakeContainers()

    fake_client = FakeClient()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def fake_wait_for(awaitable, timeout: float):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(sandbox.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(sandbox.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(sandbox.asyncio, "wait_for", fake_wait_for)

    with pytest.raises(AppError) as exc:
        asyncio.run(
            sandbox.execute_python(
                code="print('hi')",
                timeout_s=1,
                image="python:3.12-slim",
                memory_limit="256m",
            )
        )

    assert exc.value.code == "sandbox_timeout"
    assert fake_client.containers.container.killed is True
    assert fake_client.containers.container.removed is True

