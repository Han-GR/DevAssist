from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.evaluator import EvaluationSample, evaluate_sample


def test_evaluate_sample_relevance_and_faithfulness() -> None:
    sample = EvaluationSample(
        question="FastAPI 怎么做依赖注入？",
        answer="FastAPI 的依赖注入用 Depends 来声明。",
        contexts=["FastAPI 支持依赖注入，使用 Depends(...) 声明依赖。"],
    )
    r = evaluate_sample(sample)
    assert r.answer_relevance > 0.3
    assert r.faithfulness > 0.3
    assert r.context_recall is None


def test_evaluate_sample_faithfulness_zero_when_no_grounding() -> None:
    sample = EvaluationSample(
        question="FastAPI 是什么？",
        answer="它是一个前端框架，主要用于构建移动应用。",
        contexts=["FastAPI 是一个 Python Web 框架，基于 Starlette 和 Pydantic。"],
    )
    r = evaluate_sample(sample)
    assert r.faithfulness == 0.0


def test_evaluate_sample_context_recall_requires_reference_answer() -> None:
    sample = EvaluationSample(
        question="FastAPI 是什么？",
        answer="FastAPI 是 Python Web 框架。",
        contexts=["FastAPI 是一个 Python Web 框架。"],
        reference_answer="FastAPI 是一个 Python Web 框架。",
    )
    r = evaluate_sample(sample)
    assert r.context_recall is not None
    assert r.context_recall == pytest.approx(1.0)

