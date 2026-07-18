from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.llm import LLMClient
from app.finetune.eval_dataset import load_finetune_eval_cases
from app.finetune.inference import (
    InferenceConfig,
    build_chat_messages,
    format_prompt,
    generate_text,
    load_base_model_and_tokenizer,
    load_lora_model_and_tokenizer,
)
from app.finetune.judge import JudgeConfig, judge_answer


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def main_async() -> int:
    """
    LLM-as-Judge 评测脚本：对指定模型（base 或 LoRA adapter）跑 evalset，并输出 JSON 报告。

    Args:
        无（从命令行解析参数）。

    Returns:
        进程退出码。

    Raises:
        Exception: 推理或 Judge 失败时抛出。
    """

    parser = argparse.ArgumentParser(description="Run LLM-as-Judge evaluation for base or LoRA model.")
    parser.add_argument("--evalset", type=Path, default=Path("data/datasets/finetune_eval.sample.jsonl"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval_reports/judge_report.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-provider", type=str, default="deepseek")
    parser.add_argument("--judge-model", type=str, default=None, help="Override judge model if needed.")
    args = parser.parse_args()

    cases = load_finetune_eval_cases(args.evalset)[: int(args.limit)]

    infer_cfg = InferenceConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
    )

    if args.adapter:
        model, tokenizer = load_lora_model_and_tokenizer(
            base_model_name_or_path=str(args.base_model),
            adapter_path=str(args.adapter),
            config=infer_cfg,
        )
        target_label = "lora"
    else:
        model, tokenizer = load_base_model_and_tokenizer(model_name_or_path=str(args.base_model), config=infer_cfg)
        target_label = "base"

    llm_client = LLMClient(provider=str(args.judge_provider))

    judge_cfg = JudgeConfig(model=str(args.judge_model) if args.judge_model else None)

    rows: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for c in cases:
        messages = build_chat_messages(instruction=c.instruction, input_text=c.input)
        prompt = format_prompt(tokenizer=tokenizer, messages=messages)
        answer = generate_text(model=model, tokenizer=tokenizer, prompt=prompt, config=infer_cfg)
        jr = await judge_answer(llm_client=llm_client, case=c, answer=answer, config=judge_cfg)
        rows.append(
            {
                "id": c.id,
                "category": c.category,
                "score": jr.score,
                "passed": jr.passed,
                "reasons": jr.reasons,
            }
        )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    avg_score = sum(float(r["score"]) for r in rows) / max(1, len(rows))
    pass_rate = sum(1 for r in rows if bool(r["passed"])) / max(1, len(rows))

    report = {
        "target": target_label,
        "evalset": str(args.evalset),
        "count": len(rows),
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "elapsed_ms": elapsed_ms,
        "rows": rows,
    }

    _write_json(args.output_json, report)
    print(json.dumps({"avg_score": avg_score, "pass_rate": pass_rate, "count": len(rows)}, ensure_ascii=False))
    print(f"output_json={args.output_json}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

