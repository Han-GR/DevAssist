from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def _normalize_base_url_for_display(host: str, port: int) -> str:
    """
    生成 OpenAI 兼容服务的 base_url（用于输出提示）。

    Args:
        host: 服务监听地址。
        port: 服务监听端口。

    Returns:
        OpenAI client 需要的 base_url（以 /v1 结尾）。
    """

    return f"http://{host}:{port}/v1"


def build_vllm_serve_command(
    *,
    base_model: str,
    host: str,
    port: int,
    api_key: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    enable_lora: bool,
    lora_name: str | None,
    lora_path: Path | None,
    max_loras: int,
    max_lora_rank: int,
) -> list[str]:
    """
    构造 vLLM OpenAI 兼容服务的启动命令（支持 LoRA）。

    Args:
        base_model: HuggingFace model id 或本地模型路径。
        host: 服务监听地址。
        port: 服务监听端口。
        api_key: vLLM OpenAI 兼容接口的 API key（用于简单鉴权）。
        dtype: vLLM dtype 参数（例如 auto/float16/bfloat16）。
        tensor_parallel_size: 张量并行大小（多卡时设置 > 1）。
        gpu_memory_utilization: GPU 显存使用率（0~1）。
        trust_remote_code: 是否开启 trust_remote_code（部分模型需要）。
        enable_lora: 是否启用 LoRA。
        lora_name: LoRA adapter 在 vLLM 模型列表里的名字。
        lora_path: LoRA adapter 目录路径（应包含 adapter_config.json / adapter_model.safetensors）。
        max_loras: vLLM 同时加载的 LoRA adapter 上限。
        max_lora_rank: vLLM 允许的最大 LoRA rank（需 >= 实际 adapter rank）。

    Returns:
        可直接传给 subprocess 的命令参数列表。

    Raises:
        ValueError: LoRA 参数不完整或不合法。
    """

    cmd: list[str] = [
        "vllm",
        "serve",
        str(base_model),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--dtype",
        str(dtype),
        "--api-key",
        str(api_key),
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(gpu_memory_utilization)),
    ]

    if bool(trust_remote_code):
        cmd.append("--trust-remote-code")

    if bool(enable_lora):
        if not lora_name or lora_path is None:
            raise ValueError("enable_lora=true 时必须同时提供 --lora-name 与 --lora-path")
        cmd.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{lora_name}={str(lora_path)}",
                "--max-loras",
                str(int(max_loras)),
                "--max-lora-rank",
                str(int(max_lora_rank)),
            ]
        )

    return cmd


def main() -> int:
    """
    启动 vLLM OpenAI 兼容服务，并可选加载一个 LoRA adapter。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        ValueError: 参数不合法时抛出。
        FileNotFoundError: 未找到 LoRA adapter 目录时抛出。
        OSError: vllm 可执行文件不存在或启动失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Serve a base model via vLLM (OpenAI compatible), optionally with a LoRA adapter.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default="devassist-local")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-name", type=str, default="devassist-lora")
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--max-loras", type=int, default=1)
    parser.add_argument("--max-lora-rank", type=int, default=32)

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lora_path: Path | None = Path(args.lora_path) if args.lora_path else None
    if lora_path is not None and not lora_path.exists():
        raise FileNotFoundError(str(lora_path))

    cmd = build_vllm_serve_command(
        base_model=str(args.base_model),
        host=str(args.host),
        port=int(args.port),
        api_key=str(args.api_key),
        dtype=str(args.dtype),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        trust_remote_code=bool(args.trust_remote_code),
        enable_lora=bool(args.enable_lora),
        lora_name=str(args.lora_name) if args.lora_name else None,
        lora_path=lora_path,
        max_loras=int(args.max_loras),
        max_lora_rank=int(args.max_lora_rank),
    )

    print(f"base_url={_normalize_base_url_for_display(host=str(args.host), port=int(args.port))}")
    print(f"command={shlex.join(cmd)}")
    if bool(args.dry_run):
        return 0

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

