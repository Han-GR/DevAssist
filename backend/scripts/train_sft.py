from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.sft import SFTTrainConfig, train_sft


def main() -> int:
    """
    SFT 训练脚本入口（LoRA）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 训练依赖缺失或训练过程失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Run LoRA SFT training (Transformers + PEFT + TRL).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/datasets/sft_train.jsonl"))
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/models/qwen2.5-7b-lora"))
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    cfg = SFTTrainConfig(
        model_name_or_path=str(args.model),
        train_path=Path(args.train),
        eval_path=Path(args.eval) if args.eval else None,
        output_dir=Path(args.output),
        max_seq_length=int(args.max_seq_len),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        seed=int(args.seed),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )

    out = train_sft(cfg)
    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

