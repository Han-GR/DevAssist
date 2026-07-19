from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any

import httpx


@dataclass(frozen=True)
class BenchConfig:
    base_url: str
    api_key: str
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    total_requests: int
    concurrency: int
    timeout_s: float
    output_json: Path | None
    output_md: Path | None


def normalize_openai_base_url(raw: str) -> str:
    """
    规范化 OpenAI 兼容服务的 base_url。

    Args:
        raw: 用户输入的 base_url，允许为 http(s)://host:port 或以 /v1 结尾的完整路径。

    Returns:
        以 /v1 结尾的 base_url。
    """

    s = raw.rstrip("/")
    if s.endswith("/v1"):
        return s
    return f"{s}/v1"


def percentile(values: list[float], p: float) -> float:
    """
    计算分位数（0~100），使用线性插值。

    Args:
        values: 非空的数值列表。
        p: 分位数百分比（例如 50/95/99）。

    Returns:
        分位数对应的值。

    Raises:
        ValueError: values 为空或 p 超出范围时抛出。
    """

    if not values:
        raise ValueError("values must not be empty")
    if p < 0 or p > 100:
        raise ValueError("p must be in [0, 100]")

    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])

    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


async def one_request(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[float, dict[str, Any] | None, str | None]:
    """
    执行一次 /v1/chat/completions 请求并返回延迟与 usage。

    Args:
        client: httpx client。
        base_url: OpenAI 兼容 base_url（以 /v1 结尾）。
        api_key: API key。
        model: 模型 id（base 或 LoRA adapter 名）。
        prompt: user prompt。
        max_tokens: 生成上限。
        temperature: 采样温度。

    Returns:
        (latency_ms, usage_dict_or_none, error_message_or_none)。
    """

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        obj = resp.json()
        usage = obj.get("usage")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, usage if isinstance(usage, dict) else None, None
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, None, str(e)


async def run_bench(cfg: BenchConfig) -> dict[str, Any]:
    """
    并发运行 benchmark，返回结构化统计结果。

    Args:
        cfg: benchmark 配置。

    Returns:
        可序列化为 JSON 的统计结果。
    """

    base_url = normalize_openai_base_url(cfg.base_url)
    limits = httpx.Limits(max_connections=cfg.concurrency * 2, max_keepalive_connections=cfg.concurrency * 2)
    timeout = httpx.Timeout(cfg.timeout_s)

    latencies_ms: list[float] = []
    errors: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0

    sem = asyncio.Semaphore(cfg.concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:

        async def _task() -> None:
            nonlocal total_prompt_tokens, total_completion_tokens, total_total_tokens
            async with sem:
                latency_ms, usage, err = await one_request(
                    client=client,
                    base_url=base_url,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    prompt=cfg.prompt,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
                latencies_ms.append(float(latency_ms))
                if err is not None:
                    errors.append(err)
                    return
                if usage is None:
                    return
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                if isinstance(prompt_tokens, int):
                    total_prompt_tokens += prompt_tokens
                if isinstance(completion_tokens, int):
                    total_completion_tokens += completion_tokens
                if isinstance(total_tokens, int):
                    total_total_tokens += total_tokens

        started = time.perf_counter()
        await asyncio.gather(*[_task() for _ in range(int(cfg.total_requests))])
        elapsed_s = time.perf_counter() - started

    ok = int(cfg.total_requests) - len(errors)
    fail = len(errors)
    fail_rate = fail / max(int(cfg.total_requests), 1)

    p50 = percentile(latencies_ms, 50) if latencies_ms else 0.0
    p95 = percentile(latencies_ms, 95) if latencies_ms else 0.0
    p99 = percentile(latencies_ms, 99) if latencies_ms else 0.0
    avg = statistics.mean(latencies_ms) if latencies_ms else 0.0
    stdev = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    req_per_s = ok / elapsed_s if elapsed_s > 0 else 0.0
    total_tokens_per_s = total_total_tokens / elapsed_s if elapsed_s > 0 else 0.0
    completion_tokens_per_s = total_completion_tokens / elapsed_s if elapsed_s > 0 else 0.0

    out: dict[str, Any] = {
        "config": {
            "base_url": base_url,
            "model": cfg.model,
            "total_requests": int(cfg.total_requests),
            "concurrency": int(cfg.concurrency),
            "max_tokens": int(cfg.max_tokens),
            "temperature": float(cfg.temperature),
            "timeout_s": float(cfg.timeout_s),
        },
        "result": {
            "elapsed_s": float(elapsed_s),
            "ok": int(ok),
            "fail": int(fail),
            "fail_rate": float(fail_rate),
            "latency_ms": {
                "avg": float(avg),
                "stdev": float(stdev),
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
                "min": float(min(latencies_ms)) if latencies_ms else 0.0,
                "max": float(max(latencies_ms)) if latencies_ms else 0.0,
            },
            "throughput": {
                "req_per_s": float(req_per_s),
                "completion_tokens_per_s": float(completion_tokens_per_s),
                "total_tokens_per_s": float(total_tokens_per_s),
            },
            "tokens": {
                "prompt_tokens": int(total_prompt_tokens),
                "completion_tokens": int(total_completion_tokens),
                "total_tokens": int(total_total_tokens),
            },
        },
        "errors_sample": errors[:5],
    }

    return out


def render_markdown(summary: dict[str, Any]) -> str:
    """
    把 benchmark 结果渲染成 markdown（适合贴到 README/博客）。

    Args:
        summary: run_bench 的输出。

    Returns:
        markdown 文本。
    """

    cfg = summary.get("config", {})
    res = summary.get("result", {})
    lat = (res.get("latency_ms") or {}) if isinstance(res, dict) else {}
    thr = (res.get("throughput") or {}) if isinstance(res, dict) else {}
    tok = (res.get("tokens") or {}) if isinstance(res, dict) else {}

    return "\n".join(
        [
            "## vLLM Benchmark",
            "",
            f"- base_url: `{cfg.get('base_url')}`",
            f"- model: `{cfg.get('model')}`",
            f"- total_requests: `{cfg.get('total_requests')}`",
            f"- concurrency: `{cfg.get('concurrency')}`",
            "",
            "### Latency (ms)",
            "",
            "| avg | p50 | p95 | p99 | min | max |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {lat.get('avg', 0):.2f} | {lat.get('p50', 0):.2f} | {lat.get('p95', 0):.2f} | {lat.get('p99', 0):.2f} | {lat.get('min', 0):.2f} | {lat.get('max', 0):.2f} |",
            "",
            "### Throughput",
            "",
            "| req/s | completion tok/s | total tok/s |",
            "|---:|---:|---:|",
            f"| {thr.get('req_per_s', 0):.3f} | {thr.get('completion_tokens_per_s', 0):.3f} | {thr.get('total_tokens_per_s', 0):.3f} |",
            "",
            "### Token Usage (sum)",
            "",
            "| prompt | completion | total |",
            "|---:|---:|---:|",
            f"| {int(tok.get('prompt_tokens', 0))} | {int(tok.get('completion_tokens', 0))} | {int(tok.get('total_tokens', 0))} |",
            "",
        ]
    )


def main() -> int:
    """
    vLLM benchmark 脚本入口（OpenAI 兼容 /v1/chat/completions）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。
    """

    parser = argparse.ArgumentParser(description="Benchmark a vLLM OpenAI-compatible server (chat completions).")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="devassist-local")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", type=str, default="Explain SSE in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    cfg = BenchConfig(
        base_url=str(args.base_url),
        api_key=str(args.api_key),
        model=str(args.model),
        prompt=str(args.prompt),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        total_requests=int(args.requests),
        concurrency=int(args.concurrency),
        timeout_s=float(args.timeout_s),
        output_json=Path(args.output_json) if args.output_json else None,
        output_md=Path(args.output_md) if args.output_md else None,
    )

    summary = asyncio.run(run_bench(cfg))
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)

    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(payload + "\n", encoding="utf-8")
        print(f"output_json={cfg.output_json}")

    if cfg.output_md is not None:
        md = render_markdown(summary)
        cfg.output_md.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_md.write_text(md + "\n", encoding="utf-8")
        print(f"output_md={cfg.output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

