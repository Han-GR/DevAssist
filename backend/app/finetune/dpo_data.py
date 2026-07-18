from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.finetune.cleaning import contains_secrets, score_sample
from app.finetune.eval_dataset import EvalCase
from app.finetune.eval_runner import evaluate_with_rubric


@dataclass(frozen=True)
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    meta: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """
        转为可写入 JSONL 的 dict。

        Args:
            无。

        Returns:
            dict：包含 prompt/chosen/rejected；若 meta 存在则带上 meta。

        Raises:
            无。
        """

        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        if self.meta:
            payload["meta"] = self.meta
        return payload


def build_dpo_prompt(*, instruction: str, input_text: str) -> str:
    """
    构造 DPO 数据的 prompt 字符串。

    Args:
        instruction: system 指令。
        input_text: user 输入。

    Returns:
        prompt 字符串。

    Raises:
        ValueError: instruction 或 input_text 为空。

    Notes:
        - 该 prompt 作为 DPO 的 prompt 字段存储，并用于后续训练对齐。
    """

    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if not inst:
        raise ValueError("instruction must not be empty")
    if not inp:
        raise ValueError("input_text must not be empty")

    return "\n\n".join([f"system:\n{inst}", f"user:\n{inp}"])


def parse_two_candidates(text: str) -> tuple[str, str]:
    """
    解析模型输出的两条候选回答。

    Args:
        text: 模型输出文本，期望为 JSON object：{"a": "...", "b": "..."}。

    Returns:
        (a, b) 两个候选回答。

    Raises:
        ValueError: 解析失败或缺少字段。

    Notes:
        - 为了稳定性，仅接受 JSON object；不接受 Markdown code fence。
    """

    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model output")

    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"invalid json: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("model output must be a json object")

    a = obj.get("a")
    b = obj.get("b")
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("model output must contain string fields a and b")

    a_s = a.strip()
    b_s = b.strip()
    if not a_s or not b_s:
        raise ValueError("candidate answers must not be empty")

    return a_s, b_s


def pick_preference(
    *,
    case: EvalCase,
    a: str,
    b: str,
) -> tuple[str, str, dict[str, Any]]:
    """
    从两条候选中选择更优回答（用于生成 DPO chosen/rejected）。

    Args:
        case: 评测用例（含 instruction/input/rubric）。
        a: 候选回答 A。
        b: 候选回答 B。

    Returns:
        (chosen, rejected, meta)：meta 包含两条候选的 rubric 评分与选择原因。

    Raises:
        ValueError: 任意候选为空。

    Notes:
        - 选择逻辑优先使用 rubric：
          1) passed 优先
          2) include_rate 更高优先
          3) violated_terms 更少优先
        - 若仍打平，使用启发式质量分（score_sample）作为兜底。
        - 若候选包含明显密钥模式，直接判为 rejected（安全优先）。
    """

    a_s = (a or "").strip()
    b_s = (b or "").strip()
    if not a_s or not b_s:
        raise ValueError("candidates must not be empty")

    prompt = build_dpo_prompt(instruction=case.instruction, input_text=case.input)

    a_has_secret = contains_secrets(a_s)
    b_has_secret = contains_secrets(b_s)
    if a_has_secret and not b_has_secret:
        meta = {"reason": "a_contains_secrets", "a_contains_secrets": True, "b_contains_secrets": False}
        return b_s, a_s, meta
    if b_has_secret and not a_has_secret:
        meta = {"reason": "b_contains_secrets", "a_contains_secrets": False, "b_contains_secrets": True}
        return a_s, b_s, meta

    ra = evaluate_with_rubric(case=case, answer=a_s)
    rb = evaluate_with_rubric(case=case, answer=b_s)

    def _key(r) -> tuple[int, float, int]:
        return (1 if r.passed else 0, float(r.include_rate), -len(r.violated_terms))

    ka = _key(ra)
    kb = _key(rb)

    if ka != kb:
        chosen = a_s if ka > kb else b_s
        rejected = b_s if chosen == a_s else a_s
        meta = {
            "reason": "rubric",
            "prompt": prompt,
            "a": {"passed": ra.passed, "include_rate": ra.include_rate, "violated": ra.violated_terms},
            "b": {"passed": rb.passed, "include_rate": rb.include_rate, "violated": rb.violated_terms},
        }
        return chosen, rejected, meta

    qa = score_sample(instruction=case.instruction, input=case.input, output=a_s)
    qb = score_sample(instruction=case.instruction, input=case.input, output=b_s)

    if qa != qb:
        chosen = a_s if qa > qb else b_s
        rejected = b_s if chosen == a_s else a_s
        meta = {
            "reason": "heuristic_quality",
            "prompt": prompt,
            "a_quality": qa,
            "b_quality": qb,
        }
        return chosen, rejected, meta

    chosen = a_s
    rejected = b_s
    meta = {
        "reason": "tie_breaker",
        "prompt": prompt,
        "a_quality": qa,
        "b_quality": qb,
    }
    return chosen, rejected, meta


async def generate_dpo_pair_from_case(
    *,
    llm_client: Any,
    case: EvalCase,
    temperature: float,
) -> DPOPair:
    """
    生成一条 DPO 样本（prompt + chosen + rejected）。

    Args:
        llm_client: LLMClient 或兼容对象（需提供 async chat(messages, temperature, stream=False)）。
        case: 评测用例。
        temperature: 生成候选回答时的 temperature（建议 0.7）。

    Returns:
        DPOPair。

    Raises:
        ValueError: 模型输出解析失败。
        Exception: LLM 调用失败会原样抛出。

    Notes:
        - 为降低调用次数，这里让模型一次返回两条候选（JSON：{"a": "...", "b": "..."}）。
    """

    system = "\n".join(
        [
            "You are generating preference data for DPO training.",
            "Return only JSON object. No markdown fences.",
            "Schema: {\"a\": \"...\", \"b\": \"...\"}",
            "Generate two different answers (A and B). Both should be correct and helpful.",
            "Do not include secrets, tokens, credentials, or personal data.",
        ]
    )
    user = "\n\n".join([case.instruction.strip(), case.input.strip()])

    response = await llm_client.chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=float(temperature),
        stream=False,
    )
    text = str(response.choices[0].message.content or "")
    a, b = parse_two_candidates(text)
    chosen, rejected, meta = pick_preference(case=case, a=a, b=b)

    prompt = build_dpo_prompt(instruction=case.instruction, input_text=case.input)
    merged_meta = dict(meta or {})
    merged_meta["case_id"] = case.id
    merged_meta["category"] = case.category

    return DPOPair(prompt=prompt, chosen=chosen, rejected=rejected, meta=merged_meta)

