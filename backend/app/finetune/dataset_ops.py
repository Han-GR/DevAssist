from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jsonl_head(*, input_path: Path, output_path: Path, max_lines: int) -> int:
    """
    从 JSONL 文件取前 N 条记录写入新文件（保持行内容不变）。

    Args:
        input_path: 输入 JSONL 文件路径。
        output_path: 输出 JSONL 文件路径。
        max_lines: 需要保留的最大行数（>0）。

    Returns:
        实际写入的行数。

    Raises:
        FileNotFoundError: input_path 不存在。
        ValueError: max_lines 非法。
        OSError: 文件读写失败。

    Notes:
        - 该函数不解析 JSON 内容，仅按“非空行”计数截断，适合快速做小规模训练子集。
        - 如果需要随机抽样或更复杂的切分策略，建议另写专用逻辑，避免在这里引入不透明行为。
    """

    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if written >= max_lines:
                break
            raw = line.strip()
            if not raw:
                continue
            fout.write(raw + "\n")
            written += 1
    return written


def validate_jsonl_schema_minimal(*, path: Path) -> None:
    """
    对 SFT JSONL 做最小 schema 校验（instruction/input/output 必须存在且为 string）。

    Args:
        path: JSONL 文件路径。

    Returns:
        None

    Raises:
        FileNotFoundError: path 不存在。
        ValueError: 任意一行 JSON 非法或字段类型不符合要求。
    """

    if not path.exists():
        raise FileNotFoundError(str(path))

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"invalid json at line={i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"item must be object at line={i}")
        for k in ("instruction", "input", "output"):
            v = obj.get(k)
            if not isinstance(v, str):
                raise ValueError(f"field {k} must be string at line={i}")

