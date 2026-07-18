from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.finetune.eval_dataset import EvalCase


@dataclass(frozen=True)
class RubricResult:
    include_rate: float
    missing_terms: list[str]
    violated_terms: list[str]
    passed: bool


def evaluate_with_rubric(*, case: EvalCase, answer: str) -> RubricResult:
    """
    使用 EvalCase.rubric 对生成答案做轻量规则评测。

    Args:
        case: 评测用例（包含 must_include/must_not_include）。
        answer: 模型输出文本。

    Returns:
        RubricResult：包含 include_rate、缺失项、违规项与是否通过。

    Raises:
        ValueError: answer 为空。

    Notes:
        - 这是启发式评测，目标是快速对比“是否变好”，不追求严格语义等价。
        - 匹配规则为“大小写不敏感的子串包含”。
    """

    text = (answer or "").strip()
    if not text:
        raise ValueError("answer must not be empty")

    lower = text.lower()
    must_include = case.rubric.must_include
    must_not_include = case.rubric.must_not_include

    missing: list[str] = [t for t in must_include if t.strip() and t.strip().lower() not in lower]
    violated: list[str] = [t for t in must_not_include if t.strip() and t.strip().lower() in lower]
    include_rate = 1.0 if not must_include else (len(must_include) - len(missing)) / len(must_include)
    passed = (len(missing) == 0) and (len(violated) == 0)

    return RubricResult(
        include_rate=float(include_rate),
        missing_terms=missing,
        violated_terms=violated,
        passed=bool(passed),
    )


def aggregate_rubric_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    汇总 rubric 评测结果列表，输出统计信息。

    Args:
        rows: 每条评测结果的 dict（应包含 category/include_rate/passed/violated_count 等字段）。

    Returns:
        汇总统计 dict。

    Raises:
        无。
    """

    if not rows:
        return {"count": 0, "avg_include_rate": 0.0, "pass_rate": 0.0, "violation_rate": 0.0}

    count = len(rows)
    avg_include_rate = sum(float(r.get("include_rate", 0.0)) for r in rows) / count
    pass_rate = sum(1 for r in rows if r.get("passed") is True) / count
    violation_rate = sum(1 for r in rows if int(r.get("violated_count", 0)) > 0) / count
    return {
        "count": count,
        "avg_include_rate": float(avg_include_rate),
        "pass_rate": float(pass_rate),
        "violation_rate": float(violation_rate),
    }

