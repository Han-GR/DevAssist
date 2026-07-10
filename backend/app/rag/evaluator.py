from __future__ import annotations

from dataclasses import dataclass

from app.rag.bm25 import tokenize


_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "at",
    "by",
    "from",
    "that",
    "this",
    "it",
    "怎么",
    "如何",
    "为什么",
    "请问",
    "帮我",
    "一下",
    "一个",
    "以及",
    "和",
    "与",
    "的",
    "了",
    "呢",
    "吗",
}


def _content_tokens(text: str) -> list[str]:
    tokens = tokenize(text)
    return [t for t in tokens if t and t not in _STOPWORDS]


def _ratio(*, numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


@dataclass(frozen=True)
class EvaluationSample:
    """
    一条 RAG 评测样本。

    Args:
        question (str): 用户问题。
        answer (str): 模型输出的回答。
        contexts (list[str]): 检索到的上下文列表（通常是 chunk 文本）。
        reference_answer (str | None): 参考答案（可选）；用于计算 Context Recall。

    Returns:
        None

    Raises:
        None

    Notes/Examples:
        这个结构刻意做得很轻：先把“可跑通的指标计算”落地，后续再扩展更严格的评测字段。
    """

    question: str
    answer: str
    contexts: list[str]
    reference_answer: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """
    一条评测结果。

    Args:
        faithfulness (float): 回答基于上下文的程度，范围 [0, 1]。
        answer_relevance (float): 回答与问题的相关性，范围 [0, 1]。
        context_recall (float | None): 上下文对参考答案的覆盖率，范围 [0, 1]；无参考答案时为 None。

    Returns:
        None

    Raises:
        None

    Notes/Examples:
        这些指标是“可解释、可复现”的启发式实现，主要用于早期调参和回归对比。
    """

    faithfulness: float
    answer_relevance: float
    context_recall: float | None


def evaluate_sample(sample: EvaluationSample) -> EvaluationResult:
    """
    计算一条样本的评测指标。

    Args:
        sample (EvaluationSample): 待评测样本。

    Returns:
        EvaluationResult: 指标计算结果。

    Raises:
        ValueError: question 为空时抛出。

    Notes/Examples:
        - Faithfulness：回答 token 是否能在 contexts 中找到（粗略近似“是否有依据”）。
        - Answer Relevance：问题 token 是否在回答中出现（粗略近似“是否答到点上”）。
        - Context Recall：参考答案 token 是否被 contexts 覆盖（近似“检索是否把关键资料找回来”）。
    """
    if not sample.question.strip():
        raise ValueError("question is required")

    q_tokens = set(_content_tokens(sample.question))
    a_tokens = set(_content_tokens(sample.answer))
    ctx_tokens = set(_content_tokens("\n".join(sample.contexts or [])))

    relevance = _ratio(numerator=len(q_tokens & a_tokens), denominator=len(q_tokens))
    faithfulness = _ratio(numerator=len(a_tokens & ctx_tokens), denominator=len(a_tokens))

    context_recall: float | None = None
    if sample.reference_answer is not None:
        ref_tokens = set(_content_tokens(sample.reference_answer))
        context_recall = _ratio(numerator=len(ref_tokens & ctx_tokens), denominator=len(ref_tokens))

    return EvaluationResult(
        faithfulness=faithfulness,
        answer_relevance=relevance,
        context_recall=context_recall,
    )

