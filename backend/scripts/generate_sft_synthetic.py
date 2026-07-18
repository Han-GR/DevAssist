from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.synthetic import SyntheticGenConfig, generate_synthetic_sft_dataset


async def _run() -> int:
    """
    合成 SFT 数据集生成脚本（async）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: LLM 调用、解析或文件写入失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Generate synthetic SFT dataset via LLM (JSONL).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/sft_synth.jsonl"),
        help="Output JSONL path (default: data/datasets/sft_synth.jsonl).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of samples to generate (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Max samples written per request (default: 5).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a senior software engineer. Answer concisely and accurately.",
        help="SFT instruction/system prompt written into each sample.",
    )
    parser.add_argument("--topic", type=str, default=None, help="Optional topic hint (e.g. FastAPI, RAG, Agent).")
    args = parser.parse_args()

    cfg = SyntheticGenConfig(
        count=int(args.count),
        batch_size=int(args.batch_size),
        temperature=float(args.temperature),
        seed=int(args.seed) if args.seed is not None else None,
    )
    written = await generate_synthetic_sft_dataset(
        output_path=args.output,
        instruction=str(args.instruction),
        config=cfg,
        topic_hint=str(args.topic) if args.topic else None,
    )
    print(f"written={written} output={args.output}")
    return 0


def main() -> int:
    """
    脚本 main。

    Args:
        无。

    Returns:
        进程退出码。
    """

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

