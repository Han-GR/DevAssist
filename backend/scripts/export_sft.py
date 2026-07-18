from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.data_pipeline import export_sft_jsonl_from_db


def _parse_dt(value: str | None) -> datetime | None:
    """
    解析 ISO-8601 时间字符串（用于过滤会话范围）。

    Args:
        value: ISO-8601 字符串，例如 `2026-07-13T08:00:00`；None 或空串表示不限制。

    Returns:
        datetime 对象；若 value 为空则返回 None。

    Raises:
        ValueError: value 非法，无法被 `datetime.fromisoformat` 解析。

    Notes:
        - 该函数不做时区补全；建议使用带时区的 ISO 字符串。
    """
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    return datetime.fromisoformat(v)


async def _run() -> int:
    """
    导出 SFT 数据集的脚本入口（async）。

    Args:
        无（从命令行解析）。

    Returns:
        进程退出码。

    Raises:
        Exception: 参数非法、数据库连接失败或文件写入失败时抛出。
    """
    parser = argparse.ArgumentParser(description="Export SFT dataset from DevAssist chat logs (JSONL).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/sft_train.jsonl"),
        help="Output JSONL path (default: data/datasets/sft_train.jsonl)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a senior software engineer. Answer concisely and accurately.",
        help="SFT instruction/system prompt.",
    )
    parser.add_argument("--conversation-limit", type=int, default=None, help="Max number of conversations to export.")
    parser.add_argument("--no-meta", action="store_true", help="Do not include meta fields in samples.")
    parser.add_argument("--since", type=str, default=None, help="Export conversations created_at >= since (ISO-8601).")
    parser.add_argument("--until", type=str, default=None, help="Export conversations created_at <= until (ISO-8601).")
    args = parser.parse_args()

    total = await export_sft_jsonl_from_db(
        output_path=args.output,
        instruction=args.instruction,
        conversation_limit=args.conversation_limit,
        include_meta=not args.no_meta,
        since=_parse_dt(args.since),
        until=_parse_dt(args.until),
    )
    print(f"exported_samples={total} output={args.output}")
    return 0


def main() -> int:
    """
    脚本 main。

    Args:
        无。

    Returns:
        进程退出码。
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
