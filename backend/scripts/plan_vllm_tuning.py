from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.versioning import detect_git_commit_short  # noqa: E402


@dataclass(frozen=True)
class PlanConfig:
    base_model: str
    api_key: str
    host: str
    port: int
    enable_lora: bool
    lora_name: str
    lora_path: Path | None
    tensor_parallel_size: int
    dtype: str
    max_model_len: int | None
    enable_chunked_prefill: bool
    max_num_seqs_list: list[int]
    max_num_batched_tokens_list: list[int]
    gpu_memory_utilizations: list[float]
    bench_requests: int
    bench_concurrency: int
    bench_timeout_s: float
    out_dir: Path


def normalize_openai_base_url(host: str, port: int) -> str:
    """
    生成 OpenAI 兼容服务 base_url。

    Args:
        host: 服务监听地址。
        port: 服务监听端口。

    Returns:
        以 /v1 结尾的 base_url。
    """

    return f"http://{host}:{port}/v1"


def parse_int_list(raw: str) -> list[int]:
    """
    解析逗号分隔的 int 列表。

    Args:
        raw: 例如 "64,128,256"。

    Returns:
        int 列表。
    """

    xs: list[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        xs.append(int(s))
    return xs


def parse_float_list(raw: str) -> list[float]:
    """
    解析逗号分隔的 float 列表。

    Args:
        raw: 例如 "0.85,0.9,0.95"。

    Returns:
        float 列表。
    """

    xs: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        xs.append(float(s))
    return xs


def build_serve_cmd(
    *,
    cfg: PlanConfig,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> list[str]:
    """
    构造 serving 命令（dry-run 形式，打印最终 vllm serve 参数）。

    Args:
        cfg: plan 配置。
        gpu_memory_utilization: GPU 显存使用率。
        max_num_seqs: 并发序列上限。
        max_num_batched_tokens: 单步 batch token 上限。

    Returns:
        命令参数列表。
    """

    cmd: list[str] = [
        "python3",
        "scripts/serve_vllm_lora.py",
        "--base-model",
        cfg.base_model,
        "--host",
        cfg.host,
        "--port",
        str(int(cfg.port)),
        "--api-key",
        cfg.api_key,
        "--dtype",
        cfg.dtype,
        "--tensor-parallel-size",
        str(int(cfg.tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(gpu_memory_utilization)),
        "--max-num-seqs",
        str(int(max_num_seqs)),
        "--max-num-batched-tokens",
        str(int(max_num_batched_tokens)),
        "--disable-log-requests",
        "--dry-run",
    ]

    if cfg.max_model_len is not None:
        cmd.extend(["--max-model-len", str(int(cfg.max_model_len))])
    if bool(cfg.enable_chunked_prefill):
        cmd.append("--enable-chunked-prefill")

    if bool(cfg.enable_lora):
        cmd.extend(["--enable-lora", "--lora-name", cfg.lora_name])
        if cfg.lora_path is not None:
            cmd.extend(["--lora-path", str(cfg.lora_path)])

    return cmd


def build_bench_cmd(
    *,
    cfg: PlanConfig,
    model: str,
    output_json: Path,
    output_md: Path,
) -> list[str]:
    """
    构造 benchmark 命令。

    Args:
        cfg: plan 配置。
        model: bench 使用的 model（base 或 LoRA 名）。
        output_json: JSON 输出路径。
        output_md: Markdown 输出路径。

    Returns:
        命令参数列表。
    """

    base_url = normalize_openai_base_url(cfg.host, cfg.port)
    return [
        "python3",
        "scripts/bench_vllm.py",
        "--base-url",
        base_url,
        "--api-key",
        cfg.api_key,
        "--model",
        model,
        "--requests",
        str(int(cfg.bench_requests)),
        "--concurrency",
        str(int(cfg.bench_concurrency)),
        "--timeout-s",
        str(float(cfg.bench_timeout_s)),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]


def main() -> int:
    """
    生成 vLLM 性能调参计划（命令清单 + manifest），默认不执行。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        FileNotFoundError: LoRA 路径不存在时抛出。
    """

    parser = argparse.ArgumentParser(description="Generate a tuning plan for vLLM serve + benchmark (prints commands, writes a manifest).")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default="devassist-local")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true")

    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-name", type=str, default="devassist-lora")
    parser.add_argument("--lora-path", type=Path, default=None)

    parser.add_argument("--max-num-seqs", type=str, default="64,128,256")
    parser.add_argument("--max-num-batched-tokens", type=str, default="8192,16384")
    parser.add_argument("--gpu-memory-utilization", type=str, default="0.85,0.90,0.95")

    parser.add_argument("--bench-requests", type=int, default=50)
    parser.add_argument("--bench-concurrency", type=int, default=5)
    parser.add_argument("--bench-timeout-s", type=float, default=120.0)

    parser.add_argument("--out-dir", type=Path, default=Path("data/eval_reports/vllm_tuning"))
    args = parser.parse_args()

    lora_path: Path | None = Path(args.lora_path) if args.lora_path else None
    if lora_path is not None and not lora_path.exists():
        raise FileNotFoundError(str(lora_path))

    cfg = PlanConfig(
        base_model=str(args.base_model),
        api_key=str(args.api_key),
        host=str(args.host),
        port=int(args.port),
        enable_lora=bool(args.enable_lora),
        lora_name=str(args.lora_name),
        lora_path=lora_path,
        tensor_parallel_size=int(args.tensor_parallel_size),
        dtype=str(args.dtype),
        max_model_len=int(args.max_model_len) if args.max_model_len is not None else None,
        enable_chunked_prefill=bool(args.enable_chunked_prefill),
        max_num_seqs_list=parse_int_list(str(args.max_num_seqs)),
        max_num_batched_tokens_list=parse_int_list(str(args.max_num_batched_tokens)),
        gpu_memory_utilizations=parse_float_list(str(args.gpu_memory_utilization)),
        bench_requests=int(args.bench_requests),
        bench_concurrency=int(args.bench_concurrency),
        bench_timeout_s=float(args.bench_timeout_s),
        out_dir=Path(args.out_dir),
    )

    commit = detect_git_commit_short(repo_root=Path(__file__).resolve().parents[1]) or "unknown"
    stamp = time.strftime("%Y%m%d")
    plan_id = f"{stamp}-{commit}"
    out_dir = cfg.out_dir / plan_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    rows: list[dict[str, object]] = []

    for gm in cfg.gpu_memory_utilizations:
        for mns in cfg.max_num_seqs_list:
            for mnbt in cfg.max_num_batched_tokens_list:
                key = f"gm{gm}-mns{mns}-mnbt{mnbt}"
                serve_cmd = build_serve_cmd(cfg=cfg, gpu_memory_utilization=gm, max_num_seqs=mns, max_num_batched_tokens=mnbt)

                base_json = out_dir / f"bench.base.{key}.json"
                base_md = out_dir / f"bench.base.{key}.md"
                base_bench_cmd = build_bench_cmd(cfg=cfg, model=cfg.base_model if not cfg.enable_lora else cfg.base_model, output_json=base_json, output_md=base_md)

                lora_bench_cmd: list[str] | None = None
                lora_json: Path | None = None
                lora_md: Path | None = None
                if bool(cfg.enable_lora):
                    lora_json = out_dir / f"bench.lora.{key}.json"
                    lora_md = out_dir / f"bench.lora.{key}.md"
                    lora_bench_cmd = build_bench_cmd(cfg=cfg, model=cfg.lora_name, output_json=lora_json, output_md=lora_md)

                row: dict[str, object] = {
                    "id": key,
                    "params": {
                        "gpu_memory_utilization": gm,
                        "max_num_seqs": mns,
                        "max_num_batched_tokens": mnbt,
                        "max_model_len": cfg.max_model_len,
                        "enable_chunked_prefill": cfg.enable_chunked_prefill,
                    },
                    "serve_cmd": shlex.join(serve_cmd),
                    "bench_base_cmd": shlex.join(base_bench_cmd),
                    "bench_lora_cmd": shlex.join(lora_bench_cmd) if lora_bench_cmd is not None else None,
                    "outputs": {
                        "bench_base_json": str(base_json),
                        "bench_base_md": str(base_md),
                        "bench_lora_json": str(lora_json) if lora_json is not None else None,
                        "bench_lora_md": str(lora_md) if lora_md is not None else None,
                    },
                }
                rows.append(row)

    manifest_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    md_lines = [
        "# vLLM Tuning Plan",
        "",
        f"- plan_id: `{plan_id}`",
        f"- base_url: `{normalize_openai_base_url(cfg.host, cfg.port)}`",
        f"- base_model: `{cfg.base_model}`",
        f"- enable_lora: `{cfg.enable_lora}`",
        f"- lora_name: `{cfg.lora_name}`" if cfg.enable_lora else "",
        "",
        "## Runs",
        "",
        "| id | gpu_mem | max_num_seqs | max_num_batched_tokens | base p50/p95/p99 | base req/s | lora p50/p95/p99 | lora req/s |",
        "|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for r in rows:
        p = (r.get("params") or {}) if isinstance(r.get("params"), dict) else {}
        md_lines.append(
            f"| {r.get('id')} | {p.get('gpu_memory_utilization')} | {p.get('max_num_seqs')} | {p.get('max_num_batched_tokens')} |  |  |  |  |"
        )

    md_path = out_dir / "plan.md"
    md_path.write_text("\n".join([x for x in md_lines if x != ""]) + "\n", encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(f"manifest={manifest_path}")
    print(f"plan_md={md_path}")

    print("")
    print("Example (print a serve command):")
    if rows:
        print(rows[0]["serve_cmd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

