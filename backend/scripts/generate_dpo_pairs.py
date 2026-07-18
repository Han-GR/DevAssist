from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.llm import LLMClient
from app.finetune.dpo_data import generate_dpo_pair_from_case
from app.finetune.eval_dataset import load_finetune_eval_cases


async def _run() -> int:
    """
    生成 DPO 偏好数据（JSONL）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: LLM 调用失败、输出解析失败或文件写入失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Generate DPO preference pairs (prompt/chosen/rejected) via LLM.")
    parser.add_argument("--evalset", type=Path, default=Path("data/datasets/finetune_eval.sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/datasets/dpo_pairs.jsonl"))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = load_finetune_eval_cases(args.evalset)
    if int(args.count) < len(cases):
        cases = cases[: int(args.count)]
    else:
        cases = cases[: len(cases)]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    client = LLMClient.from_settings(settings)

    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for c in cases:
            pair = await generate_dpo_pair_from_case(llm_client=client, case=c, temperature=float(args.temperature))
            f.write(json.dumps(pair.to_json(), ensure_ascii=False) + "\n")
            written += 1

    print(f"written={written} output={args.output}")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

