from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.dpo import DPOTrainConfig, train_dpo
from app.finetune.versioning import build_model_version_spec


def main() -> int:
    """
    DPO 训练脚本入口（LoRA）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 训练依赖缺失或训练过程失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Run LoRA DPO training (TRL DPOTrainer).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--pairs", type=Path, default=Path("data/datasets/dpo_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/models/qwen2.5-7b-dpo-lora"))
    parser.add_argument("--versioned-output", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("data/models"))
    parser.add_argument("--init-adapter", type=Path, default=None, help="Optional LoRA adapter path to continue from.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--report-to", type=str, default="", help="Comma-separated, e.g. 'wandb' or 'tensorboard'.")
    parser.add_argument("--run-name", type=str, default="dpo")
    args = parser.parse_args()

    report_to = tuple([x.strip() for x in str(args.report_to).split(",") if x.strip()])

    output_dir = Path(args.output)
    if bool(args.versioned_output):
        repo_root = Path(__file__).resolve().parents[1]
        spec = build_model_version_spec(
            base_model=str(args.model),
            variant="dpo-lora",
            repo_root=repo_root,
        )
        output_dir = Path(args.output_root) / f"{spec.base_model}-{spec.variant}-{spec.date}-{spec.commit}"

    cfg = DPOTrainConfig(
        model_name_or_path=str(args.model),
        dpo_pairs_path=Path(args.pairs),
        output_dir=output_dir,
        init_adapter_path=Path(args.init_adapter) if args.init_adapter else None,
        max_seq_length=int(args.max_seq_len),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        beta=float(args.beta),
        seed=int(args.seed),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        report_to=report_to,
        run_name=str(args.run_name) if args.run_name else None,
    )

    out = train_dpo(cfg)
    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
