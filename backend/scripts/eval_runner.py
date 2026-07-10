from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.rag.embedder import Embedder
from app.rag.evaluator import EvaluationSample, evaluate_sample
import app.rag.generator as rag_generator
from app.rag.reranker import rerank
from app.rag.retriever import hybrid_search


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


async def _run(
    *,
    items: list[_EvalItem],
    top_k: int,
    candidate_multiplier: int,
    rerank_min_score: float,
    no_llm: bool,
    collection_name: str | None,
    embedding_metrics: bool,
    output: Path | None,
) -> int:
    results: list[dict[str, Any]] = []

    for item in items:
        if no_llm:
            answer = item.answer or item.reference_answer
            if answer is None:
                raise ValueError("no_llm=true requires dataset item.answer or item.reference_answer")

            contexts = item.contexts
            if contexts is None:
                candidate_k = max(top_k * candidate_multiplier, top_k)
                candidates = await hybrid_search(
                    query=item.question,
                    top_k=candidate_k,
                    collection_name=collection_name or item.collection_name,
                )
                picked = rerank(
                    query=item.question,
                    chunks=candidates,
                    top_k=top_k,
                    min_score=rerank_min_score,
                )
                contexts = [p.content for p in picked]

            sample = EvaluationSample(
                question=item.question,
                answer=answer,
                contexts=contexts,
                reference_answer=item.reference_answer,
            )
            r = evaluate_sample(sample)
            results.append({"item": item, "answer": answer, "result": r, "contexts": contexts})
            continue

        rag_answer = await rag_generator.generate_answer(
            query=item.question,
            top_k=top_k,
            collection_name=collection_name or item.collection_name,
            candidate_multiplier=candidate_multiplier,
            rerank_min_score=rerank_min_score,
        )
        contexts = [c.content for c in rag_answer.citations]
        sample = EvaluationSample(
            question=item.question,
            answer=rag_answer.answer,
            contexts=contexts,
            reference_answer=item.reference_answer,
        )
        r = evaluate_sample(sample)
        results.append({"item": item, "answer": rag_answer.answer, "result": r, "contexts": contexts})

    if embedding_metrics:
        settings = get_settings()
        embedder = Embedder.from_settings(settings)

        def _cosine(a: list[float], b: list[float]) -> float:
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na <= 0 or nb <= 0:
                return 0.0
            return float(dot / (na * nb))

        embed_inputs: list[str] = []
        index_map: list[tuple[int, int, int, int | None]] = []
        for x in results:
            item = x["item"]
            q = (item.question or "").strip() or " "
            a = (x["answer"] or "").strip() or " "
            ctx = "\n".join(x["contexts"] or []).strip() or " "
            r = item.reference_answer

            q_idx = len(embed_inputs)
            embed_inputs.append(q)
            a_idx = len(embed_inputs)
            embed_inputs.append(a)
            ctx_idx = len(embed_inputs)
            embed_inputs.append(ctx)
            r_idx: int | None = None
            if r is not None:
                r_idx = len(embed_inputs)
                embed_inputs.append(r.strip() or " ")
            index_map.append((q_idx, a_idx, ctx_idx, r_idx))

        vectors = await embedder.embed_texts(embed_inputs)
        for x, (q_idx, a_idx, ctx_idx, r_idx) in zip(results, index_map, strict=True):
            faith = _cosine(vectors[a_idx], vectors[ctx_idx])
            rel = _cosine(vectors[q_idx], vectors[a_idx])
            rec = _cosine(vectors[r_idx], vectors[ctx_idx]) if r_idx is not None else None
            x["embedding"] = {
                "faithfulness": faith,
                "answer_relevance": rel,
                "context_recall": rec,
            }

    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    faithfulness_vals = [x["result"].faithfulness for x in results]
    relevance_vals = [x["result"].answer_relevance for x in results]
    recall_vals = [x["result"].context_recall for x in results if x["result"].context_recall is not None]

    for i, x in enumerate(results, start=1):
        item = x["item"]
        r = x["result"]
        recall_str = "-" if r.context_recall is None else f"{r.context_recall:.3f}"
        print(
            f"{i:02d} faithfulness={r.faithfulness:.3f} relevance={r.answer_relevance:.3f} context_recall={recall_str} | {item.question}"
        )

    print("")
    summary = {
        "count": len(results),
        "avg_faithfulness": _avg(faithfulness_vals),
        "avg_relevance": _avg(relevance_vals),
        "avg_context_recall": _avg(recall_vals) if recall_vals else None,
        "top_k": top_k,
        "candidate_multiplier": candidate_multiplier,
        "rerank_min_score": rerank_min_score,
        "no_llm": no_llm,
        "collection_name": collection_name,
    }

    print(f"avg faithfulness={summary['avg_faithfulness']:.3f}")
    print(f"avg relevance={summary['avg_relevance']:.3f}")
    if summary["avg_context_recall"] is None:
        print("avg context_recall=-")
    else:
        print(f"avg context_recall={summary['avg_context_recall']:.3f}")

    if embedding_metrics:
        e_faith = [x["embedding"]["faithfulness"] for x in results]
        e_rel = [x["embedding"]["answer_relevance"] for x in results]
        e_rec = [x["embedding"]["context_recall"] for x in results if x["embedding"]["context_recall"] is not None]
        summary["embedding"] = {
            "avg_faithfulness": _avg(e_faith),
            "avg_relevance": _avg(e_rel),
            "avg_context_recall": _avg(e_rec) if e_rec else None,
        }
        print("")
        print(f"avg emb_faithfulness={summary['embedding']['avg_faithfulness']:.3f}")
        print(f"avg emb_relevance={summary['embedding']['avg_relevance']:.3f}")
        if summary["embedding"]["avg_context_recall"] is None:
            print("avg emb_context_recall=-")
        else:
            print(f"avg emb_context_recall={summary['embedding']['avg_context_recall']:.3f}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "items": [
                {
                    "question": x["item"].question,
                    "collection_name": x["item"].collection_name,
                    "faithfulness": x["result"].faithfulness,
                    "answer_relevance": x["result"].answer_relevance,
                    "context_recall": x["result"].context_recall,
                    "embedding": x.get("embedding"),
                    "contexts": x["contexts"],
                }
                for x in results
            ],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--rerank-min-score", type=float, default=0.0)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--embedding-metrics", action="store_true")
    args = parser.parse_args()

    items = _load_dataset(Path(args.dataset))
    if args.limit is not None:
        items = items[: args.limit]

    output = Path(args.output) if args.output else None
    return asyncio.run(
        _run(
            items=items,
            top_k=args.top_k,
            candidate_multiplier=args.candidate_multiplier,
            rerank_min_score=args.rerank_min_score,
            no_llm=args.no_llm,
            collection_name=args.collection,
            embedding_metrics=args.embedding_metrics,
            output=output,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
