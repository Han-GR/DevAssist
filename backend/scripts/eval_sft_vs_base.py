from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.eval_dataset import load_finetune_eval_cases
from app.finetune.eval_runner import aggregate_rubric_results, evaluate_with_rubric
from app.finetune.inference import (
    InferenceConfig,
    build_chat_messages,
    format_prompt,
    generate_text,
    load_base_model_and_tokenizer,
    load_lora_model_and_tokenizer,
)


def _write_markdown(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    """
    对比 base 模型与 LoRA SFT 模型在 finetune_eval 集上的表现（轻量 rubric 规则评测）。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 模型加载或推理失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Compare base vs SFT (LoRA) on finetune evalset.")
    parser.add_argument("--evalset", type=Path, default=Path("data/datasets/finetune_eval.sample.jsonl"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter directory (optional).")
    parser.add_argument("--output-md", type=Path, default=Path("data/eval_reports/finetune_sft_vs_base_report.md"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval cases.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = load_finetune_eval_cases(args.evalset)
    if args.limit is not None:
        cases = cases[: int(args.limit)]

    infer_cfg = InferenceConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
    )

    t0 = time.perf_counter()
    base_model, base_tokenizer = load_base_model_and_tokenizer(model_name_or_path=str(args.base_model), config=infer_cfg)
    base_load_ms = int((time.perf_counter() - t0) * 1000)

    sft_model = None
    sft_tokenizer = None
    sft_load_ms = None
    if args.adapter:
        t1 = time.perf_counter()
        sft_model, sft_tokenizer = load_lora_model_and_tokenizer(
            base_model_name_or_path=str(args.base_model),
            adapter_path=str(args.adapter),
            config=infer_cfg,
        )
        sft_load_ms = int((time.perf_counter() - t1) * 1000)

    def _run_one(model, tokenizer, label: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for c in cases:
            messages = build_chat_messages(instruction=c.instruction, input_text=c.input)
            prompt = format_prompt(tokenizer=tokenizer, messages=messages)
            text = generate_text(model=model, tokenizer=tokenizer, prompt=prompt, config=infer_cfg)
            rr = evaluate_with_rubric(case=c, answer=text)
            rows.append(
                {
                    "model": label,
                    "id": c.id,
                    "category": c.category,
                    "include_rate": rr.include_rate,
                    "passed": rr.passed,
                    "missing": rr.missing_terms,
                    "violated": rr.violated_terms,
                    "violated_count": len(rr.violated_terms),
                }
            )
        return rows

    base_rows = _run_one(base_model, base_tokenizer, "base")
    sft_rows: list[dict[str, object]] = []
    if sft_model is not None and sft_tokenizer is not None:
        sft_rows = _run_one(sft_model, sft_tokenizer, "sft_lora")

    def _by_category(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {"all": rows, "normal": [], "edge": [], "adversarial": []}
        for r in rows:
            cat = str(r.get("category") or "")
            if cat in grouped:
                grouped[cat].append(r)
        return {k: aggregate_rubric_results(v) for k, v in grouped.items()}

    base_summary = _by_category(base_rows)
    sft_summary = _by_category(sft_rows) if sft_rows else None

    md_lines: list[str] = []
    md_lines.append("# SFT vs Base Evaluation Report")
    md_lines.append("")
    md_lines.append(f"- evalset: `{args.evalset}`")
    md_lines.append(f"- cases: {len(cases)}")
    md_lines.append(f"- base_model: `{args.base_model}` (load_ms={base_load_ms})")
    md_lines.append(f"- adapter: `{args.adapter}` (load_ms={sft_load_ms})" if args.adapter else "- adapter: (none)")
    md_lines.append("")
    md_lines.append("## Summary (rubric)")
    md_lines.append("")

    def _render_table(title: str, summary: dict[str, dict[str, object]]) -> None:
        md_lines.append(f"### {title}")
        md_lines.append("")
        md_lines.append("| category | count | avg_include_rate | pass_rate | violation_rate |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for cat in ["all", "normal", "edge", "adversarial"]:
            s = summary.get(cat) or {}
            md_lines.append(
                "| "
                + cat
                + " | "
                + str(s.get("count", 0))
                + " | "
                + f"{float(s.get('avg_include_rate', 0.0)):.3f}"
                + " | "
                + f"{float(s.get('pass_rate', 0.0)):.3f}"
                + " | "
                + f"{float(s.get('violation_rate', 0.0)):.3f}"
                + " |"
            )
        md_lines.append("")

    _render_table("Base", base_summary)
    if sft_summary is not None:
        _render_table("SFT (LoRA)", sft_summary)

    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- This is a heuristic rubric-based evaluation (substring match, case-insensitive).")
    md_lines.append("- Use it to compare trends, not as a strict correctness metric.")
    md_lines.append("")

    _write_markdown(path=args.output_md, content="\n".join(md_lines))

    print(json.dumps({"base": base_summary, "sft": sft_summary}, ensure_ascii=False))
    print(f"report_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

