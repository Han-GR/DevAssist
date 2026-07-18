from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.data_versioning import create_dataset_snapshot


def _parse_inputs(value: str) -> list[Path]:
    parts = [x.strip() for x in (value or "").split(",") if x.strip()]
    return [Path(x) for x in parts]


def main() -> int:
    """
    数据集快照脚本：复制指定数据文件并写入 manifest.json 以便复现。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 任意 I/O 或参数错误会原样抛出。
    """

    parser = argparse.ArgumentParser(description="Create a dataset snapshot with sha256 manifest.")
    parser.add_argument(
        "--inputs",
        type=str,
        default="data/datasets/sft_train.jsonl,data/datasets/sft_train.cleaned.jsonl,data/datasets/dpo_pairs.jsonl,data/datasets/finetune_eval.sample.jsonl",
        help="Comma-separated file paths.",
    )
    parser.add_argument("--snapshot-root", type=Path, default=Path("data/datasets/snapshots"))
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--snapshot-id", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_files = _parse_inputs(str(args.inputs))

    out_dir = create_dataset_snapshot(
        input_files=input_files,
        repo_root=repo_root,
        snapshot_root=Path(args.snapshot_root),
        label=str(args.label) if args.label else None,
        snapshot_id=str(args.snapshot_id) if args.snapshot_id else None,
    )
    print(f"snapshot_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

