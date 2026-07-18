from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.versioning import build_model_version_spec


def _lr_token(value: float) -> str:
    s = f"{value:.0e}".replace("E", "e")
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    s = s.replace("+", "")
    return s


def _write_json(*, path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_cmd(*, argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=str(cwd), check=True)


def main() -> int:
    """
    最终 SFT 训练编排入口：固定最佳超参跑一遍全量数据，并串联离线评测生成报告。

    默认 dry-run（只打印命令与写出 run plan），需要显式 --execute 才会真正训练/评测。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="Run final LoRA SFT training with the chosen best hyperparameters.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/datasets/sft_train.cleaned.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/datasets/sft_eval.jsonl"))
    parser.add_argument("--evalset", type=Path, default=Path("data/datasets/finetune_eval.sample.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/models"))
    parser.add_argument("--reports-root", type=Path, default=Path("data/eval_reports/final"))
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tag", type=str, default="final")
    parser.add_argument("--plan-json", type=Path, default=Path("data/datasets/runs/sft_final.plan.json"))

    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    spec = build_model_version_spec(base_model=str(args.base_model), variant="lora", repo_root=repo_root)

    hp = f"{str(args.tag).strip()}-r{int(args.lora_r)}-a{int(args.lora_alpha)}-lr{_lr_token(float(args.lr))}"
    output_dir = Path(args.output_root) / f"{spec.base_model}-{spec.variant}-{spec.date}-{spec.commit}-{hp}"
    report_dir = Path(args.reports_root) / f"{spec.base_model}-{spec.date}-{spec.commit}-{hp}"

    train_cmd: list[str] = [
        sys.executable,
        "scripts/train_sft.py",
        "--model",
        str(args.base_model),
        "--train",
        str(args.train),
        "--eval",
        str(args.eval),
        "--output",
        str(output_dir),
        "--max-seq-len",
        str(int(args.max_seq_len)),
        "--batch-size",
        str(int(args.batch_size)),
        "--eval-batch-size",
        str(int(args.eval_batch_size)),
        "--grad-accum",
        str(int(args.grad_accum)),
        "--epochs",
        str(int(args.epochs)),
        "--lr",
        str(float(args.lr)),
        "--seed",
        str(int(args.seed)),
        "--lora-r",
        str(int(args.lora_r)),
        "--lora-alpha",
        str(int(args.lora_alpha)),
        "--lora-dropout",
        str(float(args.lora_dropout)),
    ]

    eval_cmd: list[str] = [
        sys.executable,
        "scripts/finetune_eval_runner.py",
        "--evalset",
        str(args.evalset),
        "--base-model",
        str(args.base_model),
        "--sft-adapter",
        str(output_dir),
        "--out-dir",
        str(report_dir),
        "--enable-rubric",
    ]
    if args.limit is not None:
        eval_cmd += ["--limit", str(int(args.limit))]

    plan = {
        "base_model": str(args.base_model),
        "train": str(args.train),
        "eval": str(args.eval),
        "evalset": str(args.evalset),
        "epochs": int(args.epochs),
        "max_seq_len": int(args.max_seq_len),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum": int(args.grad_accum),
        "seed": int(args.seed),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "learning_rate": float(args.lr),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "train_cmd": train_cmd,
        "eval_cmd": eval_cmd,
        "execute": bool(args.execute),
    }
    _write_json(path=Path(args.plan_json), obj=plan)

    print(" ".join(train_cmd))
    print(" ".join(eval_cmd))
    print(f"plan_json={args.plan_json}")
    print(f"output_dir={output_dir}")
    print(f"report_dir={report_dir}")

    if bool(args.execute):
        _run_cmd(argv=train_cmd, cwd=repo_root)
        _run_cmd(argv=eval_cmd, cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

