from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.finetune.eval_dataset import EvalCase


@dataclass(frozen=True)
class JudgeConfig:
    model: str | None = None
    temperature: float = 0.0
    max_retries: int = 2


@dataclass(frozen=True)
class JudgeResult:
    score: float
    passed: bool
    reasons: list[str]
    raw: dict[str, Any] | None = None


def build_judge_messages(*, case: EvalCase, answer: str) -> list[dict[str, str]]:
    """
    构造 LLM-as-Judge 的 messages（OpenAI-style）。

    Args:
        case: 评测用例（包含 instruction/input/rubric）。
        answer: 待评测的模型输出。

    Returns:
        messages 列表，可直接传给 LLMClient.chat。

    Raises:
        ValueError: answer 为空。

    Notes:
        - 评测输出强制要求为 JSON，便于机器解析与落盘。
        - 打分范围为 0~10，越高越好；passed 是是否达标的粗粒度判定。
    """

    text = (answer or "").strip()
    if not text:
        raise ValueError("answer must not be empty")

    system = "\n".join(
        [
            "You are a strict evaluator for an AI coding assistant.",
            "You must return ONLY a single JSON object. No markdown fences. No extra text.",
            "Schema:",
            "{",
            '  "score": 0-10 (number),',
            '  "passed": true/false,',
            '  "reasons": ["short reason 1", "short reason 2"]',
            "}",
            "",
            "Scoring rubric (high-level):",
            "- correctness and completeness for the given input",
            "- follows instruction constraints",
            "- no hallucinated claims",
            "- no secrets/tokens/credentials in output",
            "- practical and concise",
        ]
    )

    must_include = case.rubric.must_include
    must_not_include = case.rubric.must_not_include
    notes = case.rubric.notes or ""

    user = "\n\n".join(
        [
            "Instruction:",
            case.instruction,
            "",
            "User input:",
            case.input,
            "",
            "Rubric:",
            f"- must_include: {json.dumps(must_include, ensure_ascii=False)}",
            f"- must_not_include: {json.dumps(must_not_include, ensure_ascii=False)}",
            f"- notes: {notes}",
            "",
            "Candidate answer:",
            text,
        ]
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judge_response(text: str) -> dict[str, Any]:
    """
    解析 Judge 输出的 JSON object。

    Args:
        text: 模型返回的原始文本。

    Returns:
        解析后的 dict。

    Raises:
        ValueError: 解析失败或结构不符合要求。

    Notes:
        - 解析策略：
          1) 直接 json.loads
          2) 截取从第一个 '{' 到最后一个 '}' 的片段再 json.loads
    """

    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty judge output")

    def _loads(s: str) -> dict[str, Any]:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("judge output must be a json object")
        return obj

    try:
        return _loads(raw)
    except Exception:
        left = raw.find("{")
        right = raw.rfind("}")
        if 0 <= left < right:
            try:
                return _loads(raw[left : right + 1])
            except Exception as exc:
                raise ValueError(f"invalid judge json: {exc}") from exc
        raise ValueError("invalid judge json")


def to_judge_result(obj: dict[str, Any]) -> JudgeResult:
    """
    将解析后的 JSON 转为 JudgeResult。

    Args:
        obj: parse_judge_response 的返回值。

    Returns:
        JudgeResult。

    Raises:
        ValueError: 字段缺失或类型不合法。
    """

    score = obj.get("score")
    passed = obj.get("passed")
    reasons = obj.get("reasons")

    if not isinstance(score, (int, float)):
        raise ValueError("score must be number")
    if not isinstance(passed, bool):
        raise ValueError("passed must be boolean")
    if not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
        raise ValueError("reasons must be list[str]")

    score_f = float(score)
    if score_f < 0 or score_f > 10:
        raise ValueError("score must be in [0, 10]")

    clean_reasons = [x.strip() for x in reasons if x.strip()]
    return JudgeResult(score=score_f, passed=passed, reasons=clean_reasons, raw=obj)


async def judge_answer(
    *,
    llm_client: Any,
    case: EvalCase,
    answer: str,
    config: JudgeConfig,
) -> JudgeResult:
    """
    使用 LLM-as-Judge 对单条答案打分并返回结构化结果。

    Args:
        llm_client: LLMClient 或兼容对象（需提供 async chat(messages, temperature, stream=False)）。
        case: 评测用例。
        answer: 待评测的模型输出。
        config: judge 配置（temperature/重试次数等）。

    Returns:
        JudgeResult。

    Raises:
        Exception: LLM 调用失败或解析失败时抛出。
    """

    messages = build_judge_messages(case=case, answer=answer)

    last_err: Exception | None = None
    for _ in range(max(1, int(config.max_retries) + 1)):
        try:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": float(config.temperature),
                "stream": False,
            }
            if config.model:
                kwargs["model"] = str(config.model)
            resp = await llm_client.chat(**kwargs)
            content = str(resp.choices[0].message.content or "")
            obj = parse_judge_response(content)
            return to_judge_result(obj)
        except Exception as exc:
            last_err = exc
            continue

    assert last_err is not None
    raise last_err
