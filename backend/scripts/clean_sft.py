from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.cleaning import CleaningLimits, clean_jsonl_file


def _write_report(*, report_path: Path, report: object) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean SFT JSONL dataset (dedup/length/secret/quality).")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--report", type=Path, default=None, help="Optional report JSON path.")
    parser.add_argument("--min-quality", type=float, default=0.3, help="Min quality score threshold.")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication.")
    parser.add_argument("--secret-handling", type=str, default="drop", choices=["drop", "redact"])
    parser.add_argument("--no-quality-meta", action="store_true", help="Do not add quality_score into meta.")
    parser.add_argument("--instruction-max", type=int, default=512)
    parser.add_argument("--input-max", type=int, default=8000)
    parser.add_argument("--output-max", type=int, default=12000)
    parser.add_argument("--output-min", type=int, default=1)
    args = parser.parse_args()

    limits = CleaningLimits(
        instruction_max_chars=args.instruction_max,
        input_max_chars=args.input_max,
        output_max_chars=args.output_max,
        output_min_chars=args.output_min,
    )
    report = clean_jsonl_file(
        input_path=args.input,
        output_path=args.output,
        limits=limits,
        min_quality_score=float(args.min_quality),
        secret_handling=str(args.secret_handling),
        deduplicate=not args.no_dedup,
        include_quality_meta=not args.no_quality_meta,
    )

    payload = {
        "total": report.total,
        "kept": report.kept,
        "dropped": report.dropped,
        "deduped": report.deduped,
        "redacted": report.redacted,
        "reasons": report.reasons,
    }

    if args.report is not None:
        _write_report(report_path=args.report, report=payload)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

