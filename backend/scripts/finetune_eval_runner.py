from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.eval_pipeline import run_finetune_eval_pipeline


def main() -> int:
    """
    微调评测流水线入口：best-effort 运行多个评测并生成总报告。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 参数或文件写入失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Run finetune evaluation pipeline (rubric + optional judge).")
    parser.add_argument("--evalset", type=Path, default=Path("data/datasets/finetune_eval.sample.jsonl"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-adapter", type=str, default=None)
    parser.add_argument("--dpo-adapter", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval_reports"))

    parser.add_argument("--enable-rubric", action="store_true")
    parser.add_argument("--disable-rubric", action="store_true")
    parser.add_argument("--enable-judge", action="store_true")
    parser.add_argument("--judge-provider", type=str, default="deepseek")
    parser.add_argument("--judge-model", type=str, default=None)
    args = parser.parse_args()

    enable_rubric = True
    if bool(args.enable_rubric):
        enable_rubric = True
    if bool(args.disable_rubric):
        enable_rubric = False

    _, report_path = run_finetune_eval_pipeline(
        evalset=Path(args.evalset),
        base_model=str(args.base_model),
        sft_adapter=str(args.sft_adapter) if args.sft_adapter else None,
        dpo_adapter=str(args.dpo_adapter) if args.dpo_adapter else None,
        limit=int(args.limit) if args.limit is not None else None,
        out_dir=Path(args.out_dir),
        enable_rubric=enable_rubric,
        enable_judge=bool(args.enable_judge),
        judge_provider=str(args.judge_provider),
        judge_model=str(args.judge_model) if args.judge_model else None,
    )

    print(f"report_md={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

