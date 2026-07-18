from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


SecretHandling = Literal["drop", "redact"]


@dataclass(frozen=True)
class CleaningLimits:
    instruction_max_chars: int = 512
    input_max_chars: int = 8000
    output_max_chars: int = 12000
    output_min_chars: int = 1


@dataclass(frozen=True)
class CleaningDecision:
    keep: bool
    reason: str | None
    quality_score: float


@dataclass(frozen=True)
class CleaningReport:
    total: int
    kept: int
    dropped: int
    deduped: int
    redacted: int
    reasons: dict[str, int]


_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-_\.]{10,}\b", re.IGNORECASE),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
]


def _normalize_for_hash(value: str) -> str:
    return "\n".join([line.rstrip() for line in value.strip().splitlines()]).strip()


def _hash_sample(*, instruction: str, input: str, output: str) -> str:
    raw = "\n\n".join(
        [
            _normalize_for_hash(instruction),
            _normalize_for_hash(input),
            _normalize_for_hash(output),
        ]
    ).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def contains_secrets(text: str) -> bool:
    """
    检测文本中是否包含明显的密钥/令牌模式。

    Args:
        text: 待检测文本。

    Returns:
        如果命中已知密钥模式则返回 True，否则返回 False。

    Raises:
        无。

    Notes:
        - 这是启发式检测，只覆盖常见 token 形态，不保证 100%。
        - 训练数据应尽量做到“宁可误杀，也不要泄漏”。
    """

    for p in _SECRET_PATTERNS:
        if p.search(text):
            return True
    return False


def redact_secrets(text: str) -> tuple[str, int]:
    """
    将文本中命中的密钥/令牌替换为占位符。

    Args:
        text: 原始文本。

    Returns:
        (redacted_text, redacted_count)。

    Raises:
        无。

    Notes:
        - 只替换已知模式命中的片段。
    """

    redacted = text
    count = 0
    for p in _SECRET_PATTERNS:
        redacted, n = p.subn("[REDACTED]", redacted)
        count += int(n)
    return redacted, count


def score_sample(*, instruction: str, input: str, output: str) -> float:
    """
    计算样本质量分（0~1，越高越好）。

    Args:
        instruction: 系统指令。
        input: 用户输入。
        output: 目标输出。

    Returns:
        0~1 的质量分。

    Raises:
        无。

    Notes:
        - 这是“轻量可解释”的启发式打分，用于快速过滤明显低质样本。
        - 目标不是完美打分，而是提供一个可持续迭代的默认规则集。
    """

    inst = (instruction or "").strip()
    inp = (input or "").strip()
    out = (output or "").strip()

    if not inst or not inp or not out:
        return 0.0

    score = 1.0

    if len(out) < 20:
        score *= 0.5

    refusal_markers = [
        "i can't",
        "i cannot",
        "as an ai",
        "i'm sorry",
        "无法",
        "不能",
        "抱歉",
        "对不起",
    ]
    out_lower = out.lower()
    if any(x in out_lower for x in refusal_markers):
        score *= 0.4

    if inp.count("\n") > 80:
        score *= 0.7

    if len(set(inp)) <= 3 and len(inp) >= 30:
        score *= 0.5

    if inp.startswith("http://") or inp.startswith("https://"):
        score *= 0.8

    return float(max(0.0, min(1.0, score)))


def clean_sample(
    raw: dict[str, Any],
    *,
    limits: CleaningLimits,
    min_quality_score: float,
    secret_handling: SecretHandling,
    include_quality_meta: bool,
) -> tuple[dict[str, Any] | None, CleaningDecision, int]:
    """
    清洗单条 SFT 样本。

    Args:
        raw: 原始 JSON 对象。
        limits: 长度与最小输出限制。
        min_quality_score: 最低质量分阈值，低于则丢弃。
        secret_handling: 命中密钥时的处理策略：drop 或 redact。
        include_quality_meta: 是否在 meta 里附加 quality_score 与 cleaning_reason。

    Returns:
        (cleaned_or_none, decision, redacted_count)。

    Raises:
        ValueError: raw 缺少必填字段或字段类型非法。

    Notes:
        - 不在这里做“训练集/验证集划分”，只做样本层面的过滤与去风险。
    """

    instruction = raw.get("instruction")
    input_text = raw.get("input")
    output_text = raw.get("output")
    meta = raw.get("meta")

    if not isinstance(instruction, str) or not isinstance(input_text, str) or not isinstance(output_text, str):
        raise ValueError("sample fields instruction/input/output must be strings")

    inst = instruction.strip()
    inp = input_text.strip()
    out = output_text.strip()

    if len(inst) > limits.instruction_max_chars:
        d = CleaningDecision(keep=False, reason="too_long_instruction", quality_score=0.0)
        return None, d, 0
    if len(inp) > limits.input_max_chars:
        d = CleaningDecision(keep=False, reason="too_long_input", quality_score=0.0)
        return None, d, 0
    if len(out) > limits.output_max_chars:
        d = CleaningDecision(keep=False, reason="too_long_output", quality_score=0.0)
        return None, d, 0
    if len(out) < limits.output_min_chars:
        d = CleaningDecision(keep=False, reason="empty_output", quality_score=0.0)
        return None, d, 0

    combined = "\n".join([inst, inp, out])
    if contains_secrets(combined):
        if secret_handling == "drop":
            d = CleaningDecision(keep=False, reason="contains_secrets", quality_score=0.0)
            return None, d, 0
        red_inst, n1 = redact_secrets(inst)
        red_inp, n2 = redact_secrets(inp)
        red_out, n3 = redact_secrets(out)
        inst, inp, out = red_inst, red_inp, red_out
        redacted_count = n1 + n2 + n3
    else:
        redacted_count = 0

    q = score_sample(instruction=inst, input=inp, output=out)
    if q < min_quality_score:
        d = CleaningDecision(keep=False, reason="low_quality", quality_score=q)
        return None, d, redacted_count

    cleaned: dict[str, Any] = {"instruction": inst, "input": inp, "output": out}
    if meta is not None:
        if not isinstance(meta, dict):
            raise ValueError("meta must be an object when provided")
        cleaned_meta = dict(meta)
    else:
        cleaned_meta = {}

    if include_quality_meta:
        cleaned_meta["quality_score"] = q

    if cleaned_meta:
        cleaned["meta"] = cleaned_meta

    d = CleaningDecision(keep=True, reason=None, quality_score=q)
    return cleaned, d, redacted_count


def clean_jsonl_file(
    *,
    input_path: Path,
    output_path: Path,
    limits: CleaningLimits | None = None,
    min_quality_score: float = 0.3,
    secret_handling: SecretHandling = "drop",
    deduplicate: bool = True,
    include_quality_meta: bool = True,
) -> CleaningReport:
    """
    清洗一份 SFT JSONL 文件并输出清洗后的 JSONL。

    Args:
        input_path: 输入 JSONL 路径。
        output_path: 输出 JSONL 路径。
        limits: 长度限制配置；为空则使用默认值（与 README 保持一致）。
        min_quality_score: 质量分阈值。
        secret_handling: 命中密钥时的处理策略：drop 或 redact。
        deduplicate: 是否按内容哈希去重。
        include_quality_meta: 是否在 meta 中附加 quality_score。

    Returns:
        CleaningReport（计数与原因统计）。

    Raises:
        FileNotFoundError: 输入文件不存在。
        ValueError: JSONL 解析失败或字段不合法。
        OSError: 文件读写失败。
    """

    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    limits = limits or CleaningLimits()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reasons: dict[str, int] = {}
    seen: set[str] = set()
    total = 0
    kept = 0
    deduped = 0
    redacted = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("each jsonl line must be an object")

            cleaned, decision, redacted_count = clean_sample(
                raw,
                limits=limits,
                min_quality_score=min_quality_score,
                secret_handling=secret_handling,
                include_quality_meta=include_quality_meta,
            )
            redacted += redacted_count

            if cleaned is None:
                key = decision.reason or "dropped"
                reasons[key] = reasons.get(key, 0) + 1
                continue

            if deduplicate:
                fp = _hash_sample(
                    instruction=cleaned["instruction"],
                    input=cleaned["input"],
                    output=cleaned["output"],
                )
                if fp in seen:
                    deduped += 1
                    reasons["deduped"] = reasons.get("deduped", 0) + 1
                    continue
                seen.add(fp)

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    dropped = total - kept - deduped
    return CleaningReport(
        total=total,
        kept=kept,
        dropped=dropped,
        deduped=deduped,
        redacted=redacted,
        reasons=reasons,
    )

