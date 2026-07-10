from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.evaluator import EvaluationSample, evaluate_sample
import app.rag.generator as rag_generator


@dataclass(frozen=True)
class _EvalItem:
    question: str
    reference_answer: str | None
    collection_name: str | None
    answer: str | None
    contexts: list[str] | None


def _load_dataset(path: Path) -> list[_EvalItem]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("dataset json must be a list")
        items = data

    parsed: list[_EvalItem] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("dataset item must be an object")

        question = str(raw.get("question") or raw.get("query") or "").strip()
        if not question:
            raise ValueError("dataset item missing question/query")

        ref = raw.get("reference_answer") or raw.get("expected") or raw.get("answer_expected")
        reference_answer = str(ref).strip() if ref is not None else None

        col = raw.get("collection_name") or raw.get("collection")
        collection_name = str(col).strip() if col is not None and str(col).strip() else None

        ans = raw.get("answer")
        answer = str(ans).strip() if ans is not None and str(ans).strip() else None

        ctxs = raw.get("contexts")
        contexts = [str(x) for x in ctxs] if isinstance(ctxs, list) else None

        parsed.append(
            _EvalItem(
                question=question,
                reference_answer=reference_answer,
                collection_name=collection_name,
                answer=answer,
                contexts=contexts,
            )
        )

    return parsed


async def _run(*, items: list[_EvalItem], top_k: int, no_llm: bool) -> int:
    results: list[tuple[_EvalItem, Any]] = []

    for item in items:
        if no_llm:
            if item.answer is None:
                raise ValueError("no_llm=true requires dataset item.answer")
            contexts = item.contexts or []
            sample = EvaluationSample(
                question=item.question,
                answer=item.answer,
                contexts=contexts,
                reference_answer=item.reference_answer,
            )
            results.append((item, evaluate_sample(sample)))
            continue

        rag_answer = await rag_generator.generate_answer(
            query=item.question,
            top_k=top_k,
            collection_name=item.collection_name,
        )
        contexts = [c.content for c in rag_answer.citations]
        sample = EvaluationSample(
            question=item.question,
            answer=rag_answer.answer,
            contexts=contexts,
            reference_answer=item.reference_answer,
        )
        results.append((item, evaluate_sample(sample)))

    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    faithfulness_vals = [r.faithfulness for _, r in results]
    relevance_vals = [r.answer_relevance for _, r in results]
    recall_vals = [r.context_recall for _, r in results if r.context_recall is not None]

    for i, (item, r) in enumerate(results, start=1):
        recall_str = "-" if r.context_recall is None else f"{r.context_recall:.3f}"
        print(
            f"{i:02d} faithfulness={r.faithfulness:.3f} relevance={r.answer_relevance:.3f} context_recall={recall_str} | {item.question}"
        )

    print("")
    print(f"avg faithfulness={_avg(faithfulness_vals):.3f}")
    print(f"avg relevance={_avg(relevance_vals):.3f}")
    if recall_vals:
        print(f"avg context_recall={_avg(recall_vals):.3f}")
    else:
        print("avg context_recall=-")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    items = _load_dataset(Path(args.dataset))
    return asyncio.run(_run(items=items, top_k=args.top_k, no_llm=args.no_llm))


if __name__ == "__main__":
    raise SystemExit(main())

