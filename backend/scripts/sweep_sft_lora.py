from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.versioning import build_model_version_spec


def _parse_int_list(value: str) -> list[int]:
    """
    解析逗号分隔的整数列表。

    Args:
        value: 形如 "8,16,32" 的字符串。

    Returns:
        list[int]。

    Raises:
        ValueError: 解析失败或列表为空。
    """
    items: list[int] = []
    for raw in (value or "").split(","):
        s = raw.strip()
        if not s:
            continue
        items.append(int(s))
    if not items:
        raise ValueError("empty int list")
    return items


def _parse_float_list(value: str) -> list[float]:
    """
    解析逗号分隔的浮点数列表。

    Args:
        value: 形如 "5e-5,1e-4,2e-4" 的字符串。

    Returns:
        list[float]。

    Raises:
        ValueError: 解析失败或列表为空。
    """
    items: list[float] = []
    for raw in (value or "").split(","):
        s = raw.strip()
        if not s:
            continue
        items.append(float(s))
    if not items:
        raise ValueError("empty float list")
    return items


def _lr_token(value: float) -> str:
    """
    将 learning rate 格式化为可用于目录名的 token。

    Args:
        value: learning rate。

    Returns:
        目录安全的字符串，例如 "2e-4"。
    """
    s = f"{value:.0e}".replace("E", "e")
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    s = s.replace("+", "")
    return s


def _write_jsonl(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """
    以 JSONL 形式写入多行记录。

    Args:
        path: 输出路径。
        rows: 每行一条 dict。

    Returns:
        None。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_cmd(*, argv: list[str], cwd: Path) -> None:
    """
    运行一个子进程命令（流式输出到当前 stdout/stderr）。

    Args:
        argv: 进程参数列表。
        cwd: 工作目录。

    Returns:
        None。

    Raises:
        subprocess.CalledProcessError: 子进程返回非 0。
    """
    subprocess.run(argv, cwd=str(cwd), check=True)


def main() -> int:
    """
    LoRA SFT 超参数 sweep：对 (lora_r, learning_rate) 做网格组合，生成可复现的训练/评测命令。

    默认行为为 dry-run（仅打印命令并写入 manifest），避免在没有 GPU/依赖的环境里误触发训练。

    Returns:
        进程退出码。

    Raises:
        Exception: 参数解析、文件写入或子进程执行失败时抛出。
    """
    parser = argparse.ArgumentParser(description="Sweep LoRA SFT hyperparameters (r, lr) with optional execution.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/datasets/sft_train.cleaned.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/datasets/sft_eval.jsonl"))
    parser.add_argument("--evalset", type=Path, default=Path("data/datasets/finetune_eval.sample.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/models"))
    parser.add_argument("--reports-root", type=Path, default=Path("data/eval_reports/sweeps"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rs", type=str, default="8,16,32")
    parser.add_argument("--lrs", type=str, default="5e-5,1e-4,2e-4")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--manifest", type=Path, default=Path("data/datasets/sweeps/sft_lora_sweep.manifest.jsonl"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    spec = build_model_version_spec(base_model=str(args.base_model), variant="lora", repo_root=repo_root)

    lora_rs = _parse_int_list(str(args.lora_rs))
    lrs = _parse_float_list(str(args.lrs))

    planned_rows: list[dict[str, Any]] = []

    for r in lora_rs:
        for lr in lrs:
            tag = f"r{int(r)}_a{int(args.lora_alpha)}_lr{_lr_token(float(lr))}"
            output_dir = Path(args.output_root) / f"{spec.base_model}-{spec.variant}-{spec.date}-{spec.commit}-{tag}"
            report_dir = Path(args.reports_root) / f"{spec.base_model}-{spec.date}-{spec.commit}-{tag}"

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
                str(float(lr)),
                "--seed",
                str(int(args.seed)),
                "--lora-r",
                str(int(r)),
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

            row = {
                "base_model": str(args.base_model),
                "train": str(args.train),
                "eval": str(args.eval),
                "evalset": str(args.evalset),
                "epochs": int(args.epochs),
                "max_seq_len": int(args.max_seq_len),
                "batch_size": int(args.batch_size),
                "grad_accum": int(args.grad_accum),
                "seed": int(args.seed),
                "lora_r": int(r),
                "lora_alpha": int(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "learning_rate": float(lr),
                "output_dir": str(output_dir),
                "report_dir": str(report_dir),
                "train_cmd": train_cmd,
                "eval_cmd": eval_cmd,
                "status": "planned" if not bool(args.execute) else "running",
            }
            planned_rows.append(row)

            print(" ".join(train_cmd))
            print(" ".join(eval_cmd))
            print("")

            if bool(args.execute):
                _run_cmd(argv=train_cmd, cwd=repo_root)
                _run_cmd(argv=eval_cmd, cwd=repo_root)
                row["status"] = "done"

    _write_jsonl(path=Path(args.manifest), rows=planned_rows)
    print(f"manifest_jsonl={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

