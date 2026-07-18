from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyntheticGenConfig:
    count: int
    batch_size: int = 5
    temperature: float = 0.7
    seed: int | None = 42


def build_synthetic_sft_messages(*, instruction: str, topic_hint: str | None = None) -> list[dict[str, Any]]:
    """
    构造用于生成 SFT 样本的 messages（OpenAI-style）。

    Args:
        instruction: 训练用的 instruction 字段（会原样写入输出样本）。
        topic_hint: 可选的主题提示，用于让样本更集中（例如 "FastAPI"、"RAG"）。

    Returns:
        OpenAI-style messages 列表（role/content），可直接传给 LLMClient.chat。

    Raises:
        ValueError: instruction 为空。

    Notes:
        - 该函数只负责构造 prompt，不调用模型。
        - 输出格式要求模型返回 JSON 数组或 JSONL（每条含 input/output）。
    """

    inst = (instruction or "").strip()
    if not inst:
        raise ValueError("instruction must not be empty")

    topic = (topic_hint or "").strip()
    topic_line = f"Topic focus: {topic}" if topic else "Topic focus: software engineering and DevAssist features"

    system = "\n".join(
        [
            "You are generating supervised fine-tuning (SFT) data for an AI coding assistant.",
            "Return only machine-readable JSON. Do not include markdown fences.",
            "Each item must be an object with keys: input, output.",
            "Do not include any secrets, tokens, credentials, or personal data.",
            topic_line,
        ]
    )

    user = "\n".join(
        [
            "Generate a small batch of diverse high-quality samples.",
            "Requirements:",
            "- input: a realistic user request (Chinese or English is OK, but prefer Chinese user queries).",
            "- output: a precise, practical answer with steps and minimal code when helpful.",
            "- avoid overly long outputs; keep it actionable.",
            "",
            "Output format:",
            "Option A: a JSON array of objects: [{\"input\": \"...\", \"output\": \"...\"}, ...]",
            "Option B: JSON Lines: one object per line.",
        ]
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_sft_items_from_text(text: str) -> list[dict[str, str]]:
    """
    从模型输出文本解析 SFT items（input/output）。

    Args:
        text: 模型返回的原始文本（可能是 JSON 数组或 JSONL）。

    Returns:
        items 列表，每个元素为 {"input": str, "output": str}。

    Raises:
        ValueError: 无法解析出合法 items。

    Notes:
        - 解析策略：
          1) 直接 json.loads（数组或对象）
          2) 按行 json.loads（JSONL）
          3) 尝试截取从第一个 '[' 到最后一个 ']' 的片段再 json.loads
    """

    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model output")

    def _normalize_item(x: Any) -> dict[str, str] | None:
        if not isinstance(x, dict):
            return None
        inp = x.get("input")
        out = x.get("output")
        if not isinstance(inp, str) or not isinstance(out, str):
            return None
        inp_s = inp.strip()
        out_s = out.strip()
        if not inp_s or not out_s:
            return None
        return {"input": inp_s, "output": out_s}

    def _as_list(value: Any) -> list[dict[str, str]]:
        if isinstance(value, dict):
            one = _normalize_item(value)
            return [one] if one else []
        if isinstance(value, list):
            items: list[dict[str, str]] = []
            for x in value:
                one = _normalize_item(x)
                if one:
                    items.append(one)
            return items
        return []

    try:
        parsed = json.loads(raw)
        items = _as_list(parsed)
        if items:
            return items
    except Exception:
        pass

    items_jsonl: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            items_jsonl = []
            break
        one = _normalize_item(obj)
        if one:
            items_jsonl.append(one)
    if items_jsonl:
        return items_jsonl

    left = raw.find("[")
    right = raw.rfind("]")
    if 0 <= left < right:
        try:
            parsed = json.loads(raw[left : right + 1])
            items = _as_list(parsed)
            if items:
                return items
        except Exception:
            pass

    raise ValueError("failed to parse SFT items from model output")


async def generate_synthetic_sft_dataset(
    *,
    output_path: Path,
    instruction: str,
    config: SyntheticGenConfig,
    topic_hint: str | None = None,
) -> int:
    """
    使用 LLM 生成合成 SFT 数据集（JSONL）。

    Args:
        output_path: 输出 JSONL 文件路径。
        instruction: 写入每条样本的 instruction 字段。
        config: 生成参数（条数、batch_size、temperature、seed）。
        topic_hint: 可选主题提示（例如 "FastAPI"、"Agent tools"）。

    Returns:
        实际写入的样本条数。

    Raises:
        ValueError: instruction 为空或 config.count 非法。
        Exception: LLM 调用失败或输出解析失败时抛出。

    Notes:
        - 本函数依赖运行环境已配置 LLM 相关环境变量（provider/api_key/model）。
        - 为避免写出大文件污染仓库，建议输出到 data/datasets/，并按 .gitignore 管理。
        - 清洗脚本可以对生成结果做去重/过滤。
    """

    inst = (instruction or "").strip()
    if not inst:
        raise ValueError("instruction must not be empty")
    if config.count <= 0:
        raise ValueError("config.count must be > 0")
    if config.batch_size <= 0:
        raise ValueError("config.batch_size must be > 0")

    from app.core.config import get_settings
    from app.core.llm import LLMClient

    settings = get_settings()
    client = LLMClient.from_settings(settings)

    if config.seed is not None:
        random.seed(config.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        while written < config.count:
            messages = build_synthetic_sft_messages(instruction=inst, topic_hint=topic_hint)
            response = await client.chat(messages=messages, temperature=float(config.temperature), stream=False)
            content = str(response.choices[0].message.content or "")
            items = parse_sft_items_from_text(content)

            random.shuffle(items)
            remaining = config.count - written
            picked = items[: min(len(items), remaining, config.batch_size)]
            if not picked:
                raise ValueError("model returned no usable items")

            for x in picked:
                record = {"instruction": inst, "input": x["input"], "output": x["output"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            await asyncio.sleep(0.2)

    return written

