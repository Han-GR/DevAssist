from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


EvalCategory = Literal["normal", "edge", "adversarial"]


@dataclass(frozen=True)
class EvalRubric:
    must_include: list[str]
    must_not_include: list[str]
    notes: str | None = None


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: EvalCategory
    instruction: str
    input: str
    rubric: EvalRubric


def load_finetune_eval_cases(path: Path) -> list[EvalCase]:
    """
    读取微调评测集（JSONL）并解析为结构化对象。

    Args:
        path: JSONL 文件路径。

    Returns:
        EvalCase 列表。

    Raises:
        FileNotFoundError: path 不存在。
        ValueError: 行内容不是合法 JSON，或字段缺失/类型不合法。

    Notes:
        - 该评测集用于后续“规则检查”和“LLM-as-judge”评测，不等同于 SFT 的训练/验证集。
        - 每行必须是一个 JSON object，且至少包含：id/category/instruction/input/rubric。
    """

    if not path.exists():
        raise FileNotFoundError(str(path))

    cases: list[EvalCase] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw_line = line.strip()
        if not raw_line:
            continue
        try:
            raw = json.loads(raw_line)
        except Exception as exc:
            raise ValueError(f"invalid json at line={i}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"item must be object at line={i}")

        case_id = raw.get("id")
        category = raw.get("category")
        instruction = raw.get("instruction")
        input_text = raw.get("input")
        rubric = raw.get("rubric")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"missing id at line={i}")
        if category not in ("normal", "edge", "adversarial"):
            raise ValueError(f"invalid category at line={i}")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"missing instruction at line={i}")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"missing input at line={i}")
        if not isinstance(rubric, dict):
            raise ValueError(f"missing rubric at line={i}")

        must_include = rubric.get("must_include") or []
        must_not_include = rubric.get("must_not_include") or []
        notes = rubric.get("notes")

        if not isinstance(must_include, list) or not all(isinstance(x, str) for x in must_include):
            raise ValueError(f"rubric.must_include must be string list at line={i}")
        if not isinstance(must_not_include, list) or not all(isinstance(x, str) for x in must_not_include):
            raise ValueError(f"rubric.must_not_include must be string list at line={i}")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(f"rubric.notes must be string at line={i}")

        cases.append(
            EvalCase(
                id=case_id.strip(),
                category=category,
                instruction=instruction.strip(),
                input=input_text.strip(),
                rubric=EvalRubric(
                    must_include=[x.strip() for x in must_include if x.strip()],
                    must_not_include=[x.strip() for x in must_not_include if x.strip()],
                    notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
                ),
            )
        )

    return cases


def summarize_eval_cases(cases: list[EvalCase]) -> dict[str, Any]:
    """
    汇总评测集统计信息（按 category 计数）。

    Args:
        cases: EvalCase 列表。

    Returns:
        统计信息 dict，例如 {"total": 200, "normal": 120, "edge": 50, "adversarial": 30}。

    Raises:
        无。
    """

    summary: dict[str, Any] = {"total": len(cases), "normal": 0, "edge": 0, "adversarial": 0}
    for c in cases:
        summary[c.category] = int(summary.get(c.category, 0)) + 1
    return summary

