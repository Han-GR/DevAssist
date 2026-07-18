from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.eval_dataset import load_finetune_eval_cases, summarize_eval_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate finetune evalset JSONL and print summary.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/datasets/finetune_eval.sample.jsonl"),
        help="Evalset JSONL path (default: data/datasets/finetune_eval.sample.jsonl)",
    )
    args = parser.parse_args()

    cases = load_finetune_eval_cases(args.path)
    summary = summarize_eval_cases(cases)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

