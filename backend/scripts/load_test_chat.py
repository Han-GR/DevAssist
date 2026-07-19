from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any
from uuid import uuid4

import httpx

from app.finetune.versioning import detect_git_commit_short


@dataclass(frozen=True)
class LoadTestConfig:
    base_url: str
    total_requests: int
    concurrency: int
    timeout_s: float
    startup_timeout_s: float
    retry_interval_s: float
    stream: bool
    model_source: str
    model: str | None
    output_dir: Path | None


def _normalize_base_url(value: str) -> str:
    """
    规范化 base_url。

    Args:
        value (str): base_url。

    Returns:
        str: 去掉末尾斜杠后的 base_url。

    Raises:
        ValueError: base_url 为空时抛出。
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("base_url is required")
    return raw.rstrip("/")


def _percentile(values: list[float], *, p: float) -> float:
    """
    计算百分位数（p in [0, 100]）。

    Args:
        values (list[float]): 数值列表。
        p (float): 百分位（0~100）。

    Returns:
        float: 百分位数。

    Raises:
        ValueError: values 为空或 p 非法。
    """
    if not values:
        raise ValueError("values is empty")
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


def _bucketize(values_ms: list[float]) -> dict[str, int]:
    """
    把延迟按固定桶聚合，便于画直方图。

    Args:
        values_ms (list[float]): 延迟（毫秒）。

    Returns:
        dict[str, int]: 桶 -> count。
    """
    edges = [50, 100, 200, 400, 800, 1200, 2000, 4000, 8000]
    buckets: dict[str, int] = {}
    for v in values_ms:
        placed = False
        for e in edges:
            if v <= e:
                key = f"<= {e}ms"
                buckets[key] = buckets.get(key, 0) + 1
                placed = True
                break
        if not placed:
            buckets["> 8000ms"] = buckets.get("> 8000ms", 0) + 1
    return buckets


async def _wait_health(*, client: httpx.AsyncClient, cfg: LoadTestConfig) -> None:
    """
    等待服务健康。

    Args:
        client (httpx.AsyncClient): HTTP client。
        cfg (LoadTestConfig): 配置。

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
                return
            last_error = f"status={resp.status_code} body={resp.text}"
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(cfg.retry_interval_s)
    raise TimeoutError(f"health check timeout: {last_error}")


async def _one_chat(
    *,
    client: httpx.AsyncClient,
    cfg: LoadTestConfig,
    index: int,
) -> dict[str, Any]:
    """
    发起一次 /chat 请求并记录延迟。

    Args:
        client (httpx.AsyncClient): HTTP client。
        cfg (LoadTestConfig): 配置。
        index (int): 序号（用于生成 user_id）。

    Returns:
        dict[str, Any]: 结果记录。
    """
    user_id = f"loadtest-{index}"
    headers = {"x-user-id": user_id}
    payload: dict[str, Any] = {
        "message": "loadtest: ping",
        "history": [],
        "use_rag": False,
        "use_agent": False,
        "model_source": cfg.model_source,
    }
    if cfg.model is not None and cfg.model.strip():
        payload["model"] = cfg.model.strip()

    url = f"{cfg.base_url}/chat"
    params = {"stream": "true"} if cfg.stream else None

    started = time.perf_counter()
    ttfb_ms: float | None = None
    status_code: int | None = None
    error: str | None = None

    try:
        if not cfg.stream:
            resp = await client.post(url, json=payload, headers=headers)
            status_code = resp.status_code
            resp.raise_for_status()
            body = resp.json()
            if not str(body.get("reply") or "").strip():
                raise RuntimeError("empty reply")
        else:
            async with client.stream("POST", url, params=params, json=payload, headers=headers) as resp:
                status_code = resp.status_code
                resp.raise_for_status()
                async for _line in resp.aiter_lines():
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started) * 1000.0
        ok = True
    except Exception as exc:
        ok = False
        error = str(exc)

    duration_ms = (time.perf_counter() - started) * 1000.0
    return {
        "ok": ok,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "ttfb_ms": ttfb_ms,
        "error": error,
        "user_id": user_id,
    }


def _render_markdown(*, cfg: LoadTestConfig, summary: dict[str, Any]) -> str:
    """
    生成 Markdown 报告。

    Args:
        cfg (LoadTestConfig): 配置。
        summary (dict[str, Any]): 汇总结果。

    Returns:
        str: Markdown。
    """
    return "\n".join(
        [
            "# DevAssist Load Test Report",
            "",
            f"- base_url: {cfg.base_url}",
            f"- total_requests: {cfg.total_requests}",
            f"- concurrency: {cfg.concurrency}",
            f"- stream: {int(cfg.stream)}",
            f"- model_source: {cfg.model_source}",
            f"- model: {(cfg.model or '').strip() or '(default)'}",
            "",
            "## Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| ok_requests | {summary['ok_requests']} |",
            f"| error_requests | {summary['error_requests']} |",
            f"| error_rate | {summary['error_rate']:.4f} |",
            f"| total_wall_ms | {summary['total_wall_ms']:.1f} |",
            f"| req_per_s | {summary['req_per_s']:.2f} |",
            "",
            "## Latency (ms)",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| avg | {summary['latency_ms']['avg']:.1f} |",
            f"| stdev | {summary['latency_ms']['stdev']:.1f} |",
            f"| p50 | {summary['latency_ms']['p50']:.1f} |",
            f"| p95 | {summary['latency_ms']['p95']:.1f} |",
            f"| p99 | {summary['latency_ms']['p99']:.1f} |",
            f"| min | {summary['latency_ms']['min']:.1f} |",
            f"| max | {summary['latency_ms']['max']:.1f} |",
            "",
        ]
    )


def _render_html(*, cfg: LoadTestConfig, summary: dict[str, Any], latencies_ms: list[float]) -> str:
    """
    生成 HTML 报告（含 Chart.js 图表）。

    Args:
        cfg (LoadTestConfig): 配置。
        summary (dict[str, Any]): 汇总结果。
        latencies_ms (list[float]): 延迟样本。

    Returns:
        str: HTML。
    """
    payload = json.dumps(
        {
            "config": {
                "base_url": cfg.base_url,
                "total_requests": cfg.total_requests,
                "concurrency": cfg.concurrency,
                "stream": cfg.stream,
                "model_source": cfg.model_source,
                "model": cfg.model,
            },
            "summary": summary,
            "latencies_ms": latencies_ms,
            "histogram": summary.get("histogram", {}),
        },
        ensure_ascii=False,
    )

    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8" />',
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "<title>DevAssist Load Test Report</title>",
            "<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.3\"></script>",
            "<style>",
            "body{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px;}",
            "pre{background:#0b1020;color:#e5e7eb;padding:12px;border-radius:8px;overflow:auto;}",
            "canvas{max-width: 100%;}",
            ".grid{display:grid;grid-template-columns:1fr;gap:16px;}",
            "@media (min-width: 980px){.grid{grid-template-columns:1fr 1fr;}}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>DevAssist Load Test Report</h1>",
            "<div class=\"grid\">",
            "<div>",
            "<h2>Latency Series</h2>",
            "<canvas id=\"latencyChart\"></canvas>",
            "</div>",
            "<div>",
            "<h2>Latency Histogram</h2>",
            "<canvas id=\"histChart\"></canvas>",
            "</div>",
            "</div>",
            "<h2>Raw</h2>",
            "<pre id=\"raw\"></pre>",
            "<script>",
            f"const data = {payload};",
            "document.getElementById('raw').textContent = JSON.stringify(data.summary, null, 2);",
            "const labels = data.latencies_ms.map((_, i) => i + 1);",
            "new Chart(document.getElementById('latencyChart'), {",
            "  type: 'line',",
            "  data: { labels, datasets: [{ label: 'latency_ms', data: data.latencies_ms, borderColor: '#2563eb', pointRadius: 0, borderWidth: 1 }] },",
            "  options: { responsive: true, plugins: { legend: { display: true } }, scales: { x: { display: false } } }",
            "});",
            "const histLabels = Object.keys(data.histogram);",
            "const histValues = histLabels.map((k) => data.histogram[k]);",
            "new Chart(document.getElementById('histChart'), {",
            "  type: 'bar',",
            "  data: { labels: histLabels, datasets: [{ label: 'count', data: histValues, backgroundColor: '#16a34a' }] },",
            "  options: { responsive: true, plugins: { legend: { display: true } } }",
            "});",
            "</script>",
            "</body>",
            "</html>",
        ]
    )


async def run_load_test(*, cfg: LoadTestConfig) -> dict[str, Any]:
    """
    执行压测并返回结果对象。

    Args:
        cfg (LoadTestConfig): 配置。

    Returns:
        dict[str, Any]: 结果（config/summary/latencies）。
    """
    limits = httpx.Limits(max_keepalive_connections=cfg.concurrency, max_connections=cfg.concurrency)
    async with httpx.AsyncClient(timeout=cfg.timeout_s, limits=limits) as client:
        await _wait_health(client=client, cfg=cfg)

        sem = asyncio.Semaphore(cfg.concurrency)

        async def _guarded(i: int) -> dict[str, Any]:
            async with sem:
                return await _one_chat(client=client, cfg=cfg, index=i)

        started_wall = time.perf_counter()
        tasks = [asyncio.create_task(_guarded(i)) for i in range(cfg.total_requests)]
        results = await asyncio.gather(*tasks)
        total_wall_ms = (time.perf_counter() - started_wall) * 1000.0

    ok = [r for r in results if r.get("ok")]
    err = [r for r in results if not r.get("ok")]
    latencies_ms = [float(r["duration_ms"]) for r in ok]

    summary: dict[str, Any] = {
        "ok_requests": len(ok),
        "error_requests": len(err),
        "error_rate": (len(err) / max(len(results), 1)),
        "total_wall_ms": total_wall_ms,
        "req_per_s": (len(ok) / max(total_wall_ms / 1000.0, 1e-9)),
        "latency_ms": None,
        "histogram": _bucketize(latencies_ms),
    }

    if latencies_ms:
        avg = statistics.fmean(latencies_ms)
        stdev = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        summary["latency_ms"] = {
            "avg": avg,
            "stdev": stdev,
            "p50": _percentile(latencies_ms, p=50),
            "p95": _percentile(latencies_ms, p=95),
            "p99": _percentile(latencies_ms, p=99),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        }
    else:
        summary["latency_ms"] = {
            "avg": 0.0,
            "stdev": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {"config": asdict(cfg), "summary": summary, "results": results, "latencies_ms": latencies_ms}


def _default_output_dir(*, repo_root: Path) -> Path:
    """
    构造默认输出目录（带日期与 commit）。

    Args:
        repo_root (Path): 仓库根目录。

    Returns:
        Path: 输出目录。
    """
    date = datetime.now().strftime("%Y%m%d")
    commit = detect_git_commit_short(repo_root=repo_root) or "unknown"
    return repo_root / "data" / "eval_reports" / "load_test" / f"{date}-{commit}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--total-requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-interval-s", type=float, default=2.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--model-source", default="remote", choices=["remote", "local"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(repo_root=repo_root)
    cfg = LoadTestConfig(
        base_url=_normalize_base_url(str(args.base_url)),
        total_requests=int(args.total_requests),
        concurrency=int(args.concurrency),
        timeout_s=float(args.timeout_s),
        startup_timeout_s=float(args.startup_timeout_s),
        retry_interval_s=float(args.retry_interval_s),
        stream=bool(args.stream),
        model_source=str(args.model_source),
        model=str(args.model) if args.model is not None else None,
        output_dir=output_dir,
    )

    try:
        result = asyncio.run(run_load_test(cfg=cfg))
        summary = result["summary"]
        latencies_ms = result["latencies_ms"]

        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / "report.json"
        out_md = output_dir / "report.md"
        out_html = output_dir / "report.html"

        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.write_text(_render_markdown(cfg=cfg, summary=summary), encoding="utf-8")
        out_html.write_text(_render_html(cfg=cfg, summary=summary, latencies_ms=latencies_ms), encoding="utf-8")

        print(f"report_json={out_json}")
        print(f"report_md={out_md}")
        print(f"report_html={out_html}")
        print("load_test_ok=1")
        return 0
    except Exception as exc:
        print(json.dumps({"load_test_ok": 0, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

