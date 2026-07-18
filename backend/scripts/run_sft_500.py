from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.dataset_ops import validate_jsonl_schema_minimal, write_jsonl_head
from app.finetune.sft import SFTTrainConfig, train_sft


def main() -> int:
    """
    运行一个小规模 SFT 训练实验（固定抽取 500 条样本）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 数据集不存在、schema 不合法或训练过程失败时抛出。

    Notes:
        - 默认取输入 JSONL 的前 500 条非空行，便于复现与调试。
        - 如需随机抽样，请先在外部生成子集文件，再传入 --train。
        - 训练日志可通过 --report-to 透出给 transformers 支持的后端（例如 wandb/tensorboard）。
    """

    parser = argparse.ArgumentParser(description="Run an initial LoRA SFT experiment on 500 samples.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/datasets/sft_train.cleaned.jsonl"))
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/models/qwen2.5-7b-lora-500"))
    parser.add_argument("--subset", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="", help="Comma-separated, e.g. 'wandb' or 'tensorboard'.")
    parser.add_argument("--run-name", type=str, default="sft-500")
    args = parser.parse_args()

    if not args.train.exists():
        fallback = Path("data/datasets/sft_train.jsonl")
        if fallback.exists():
            args.train = fallback
        else:
            raise FileNotFoundError(str(args.train))

    subset_path = args.train.with_suffix(f".{int(args.subset)}.jsonl")
    written = write_jsonl_head(input_path=Path(args.train), output_path=subset_path, max_lines=int(args.subset))
    if written <= 0:
        raise ValueError("no samples written to subset file")

    validate_jsonl_schema_minimal(path=subset_path)

    report_to = tuple([x.strip() for x in str(args.report_to).split(",") if x.strip()])

    cfg = SFTTrainConfig(
        model_name_or_path=str(args.model),
        train_path=subset_path,
        eval_path=Path(args.eval) if args.eval else None,
        output_dir=Path(args.output),
        max_seq_length=int(args.max_seq_len),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        seed=int(args.seed),
        report_to=report_to,
        run_name=str(args.run_name) if args.run_name else None,
    )

    out = train_sft(cfg)
    print(f"subset_path={subset_path}")
    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

