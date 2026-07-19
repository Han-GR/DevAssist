from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
import sys
import time
from typing import Any
from uuid import uuid4

import httpx


@dataclass(frozen=True)
class SmokeConfig:
    base_url: str
    timeout_s: float
    startup_timeout_s: float
    retry_interval_s: float
    trust_env: bool


def _normalize_base_url(value: str) -> str:
    """
    规范化 base_url。

    Args:
        value (str): 输入的 base_url。

    Returns:
        str: 去掉末尾斜杠后的 base_url。

    Raises:
        ValueError: base_url 为空时抛出。
    """
    if not value.strip():
        raise ValueError("base_url is required")
    return value.rstrip("/")


async def _wait_health(*, client: httpx.AsyncClient, cfg: SmokeConfig) -> None:
    """
    等待服务健康。

    Args:
        client (httpx.AsyncClient): HTTP client。
        cfg (SmokeConfig): 配置。

    Returns:
        None

    Raises:
        TimeoutError: 超时未健康时抛出。
    """
    deadline = time.monotonic() + cfg.startup_timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = await client.get(f"{cfg.base_url}/health")
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                print("health_ok=1")
                return
            last_error = f"status={resp.status_code} body={resp.text}"
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(cfg.retry_interval_s)
    raise TimeoutError(f"health check timeout: {last_error}")


async def _post_json(
    *,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> httpx.Response:
    """
    发送 JSON POST。

    Args:
        client (httpx.AsyncClient): HTTP client。
        url (str): 目标 URL。
        payload (dict[str, Any]): JSON payload。

    Returns:
        httpx.Response: 响应。

    Raises:
        httpx.HTTPError: 网络层异常。
    """
    return await client.post(url, json=payload)


async def run_smoke(*, cfg: SmokeConfig) -> None:
    """
    端到端 smoke 测试入口。

    Args:
        cfg (SmokeConfig): 配置。

    Returns:
        None

    Raises:
        Exception: 任一环节失败则抛出。
    """
    headers = {"x-user-id": "smoke-user"}
    async with httpx.AsyncClient(timeout=cfg.timeout_s, headers=headers, trust_env=cfg.trust_env) as client:
        await _wait_health(client=client, cfg=cfg)

        chat_resp = await _post_json(
            client=client,
            url=f"{cfg.base_url}/chat",
            payload={"message": "smoke: ping"},
        )
        chat_resp.raise_for_status()
        chat_body = chat_resp.json()
        if not str(chat_body.get("reply", "")).strip():
            raise RuntimeError(f"/chat empty reply: {chat_body}")
        print("chat_ok=1")

        marker = f"smoke-doc-{uuid4().hex}"
        files = {
            "file": (
                "smoke.txt",
                f"hello from devassist smoke test\nmarker={marker}\n".encode("utf-8"),
                "text/plain",
            )
        }
        ingest_resp = await client.post(f"{cfg.base_url}/ingest", files=files)
        ingest_resp.raise_for_status()
        ingest_body = ingest_resp.json()
        collection = str(ingest_body.get("collection", "")).strip()
        if not collection:
            raise RuntimeError(f"/ingest missing collection: {ingest_body}")
        print("ingest_ok=1")

        search_resp = await _post_json(
            client=client,
            url=f"{cfg.base_url}/search",
            payload={"query": marker, "top_k": 3, "collection_name": collection},
        )
        search_resp.raise_for_status()
        search_body = search_resp.json()
        results = list(search_body.get("results") or [])
        if not results:
            raise RuntimeError(f"/search empty results: {search_body}")
        if not any(marker in str(r.get("content", "")) for r in results):
            raise RuntimeError("search results do not contain marker")
        print("search_ok=1")

        agent_resp = await _post_json(
            client=client,
            url=f"{cfg.base_url}/agent",
            payload={"message": "smoke: reply with a short ok message"},
        )
        agent_resp.raise_for_status()
        agent_body = agent_resp.json()
        answer = str(agent_body.get("answer", "")).strip()
        if not answer:
            raise RuntimeError(f"/agent empty answer: {agent_body}")
        print("agent_ok=1")

        print("smoke_ok=1")


def _build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    Returns:
        argparse.ArgumentParser: parser。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEVASSIST_BASE_URL", "http://127.0.0.1"),
        help="DevAssist base url (default: http://127.0.0.1)",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Whether to trust environment variables (proxy/no_proxy, etc). Default: false.",
    )
    return parser


def main() -> int:
    """
    CLI 入口。

    Returns:
        int: 进程退出码（0 成功，非 0 失败）。
    """
    args = _build_parser().parse_args()
    cfg = SmokeConfig(
        base_url=_normalize_base_url(str(args.base_url)),
        timeout_s=float(args.timeout_s),
        startup_timeout_s=float(args.startup_timeout_s),
        retry_interval_s=float(args.retry_interval_s),
        trust_env=bool(args.trust_env),
    )

    try:
        asyncio.run(run_smoke(cfg=cfg))
        return 0
    except Exception as exc:
        print(json.dumps({"smoke_ok": 0, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
