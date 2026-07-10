from __future__ import annotations

import re


def split_text(text: str, *, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    将长文本按固定长度切分为多个 chunk，并在相邻 chunk 之间保留重叠区间。

    Args:
        text (str): 待切分的原始文本。
        chunk_size (int): 每个 chunk 的最大字符数，默认 512。
        overlap (int): 相邻 chunk 的重叠字符数，默认 64。

    Returns:
        list[str]: 切分后的 chunk 列表，按原文顺序排列。

    Raises:
        ValueError: 当 chunk_size 非正数，或 overlap 为负数/不小于 chunk_size 时抛出。

    Notes/Examples:
        - 这是最基础的“定长切分”，以字符长度为准；后续做语义切分或代码块感知时，可以在此基础上扩展。
        - overlap 的作用是给检索/生成保留上下文衔接，避免关键句子刚好被切断。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])

        if end >= text_len:
            break

        start = end - overlap

    return chunks


def split_text_semantic(
    text: str, *, chunk_size: int = 512, overlap: int = 64
) -> list[str]:
    """
    将文本做“更像人读文章”的切分：尽量按句子/段落边界切分，并尽量避免拆开 Markdown 代码块。

    Args:
        text (str): 待切分的原始文本。
        chunk_size (int): 目标 chunk 长度（按字符数粗略控制），默认 512。
        overlap (int): 相邻 chunk 的重叠长度（按字符数粗略控制），默认 64。

    Returns:
        list[str]: 切分后的 chunk 列表。

    Raises:
        ValueError: 当 chunk_size 非正数，或 overlap 为负数/不小于 chunk_size 时抛出。

    Notes/Examples:
        - 语义切分并不保证每个 chunk 都严格等长：遇到较长的代码块时，会优先保持代码块完整。
        - overlap 的实现按“完整单元”拼接（句子/段落分隔/代码块），避免在句子中间或代码块中间硬切。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    units: list[str] = []
    for segment in _split_by_fenced_code_blocks(text):
        if segment.startswith("```") and segment.rstrip().endswith("```"):
            units.append(segment)
            continue
        units.extend(_split_text_segment_to_units(segment))

    chunks_units: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current and current_len + unit_len > chunk_size:
            chunks_units.append(current)
            current = []
            current_len = 0

        if not current and unit_len > chunk_size:
            chunks_units.append([unit])
            continue

        current.append(unit)
        current_len += unit_len

    if current:
        chunks_units.append(current)

    if overlap == 0:
        return _finalize_chunks(chunks_units)

    overlapped: list[list[str]] = [chunks_units[0]]
    for i in range(1, len(chunks_units)):
        prev = overlapped[-1]
        prefix: list[str] = []
        acc = 0
        for unit in reversed(prev):
            prefix.append(unit)
            acc += len(unit)
            if acc >= overlap:
                break
        prefix.reverse()
        overlapped.append(prefix + chunks_units[i])

    return _finalize_chunks(overlapped)


def _split_by_fenced_code_blocks(text: str) -> list[str]:
    in_code = False
    buffer: list[str] = []
    segments: list[str] = []

    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if not in_code:
                if buffer:
                    segments.append("".join(buffer))
                    buffer = []
                in_code = True
                buffer.append(line)
                continue

            buffer.append(line)
            segments.append("".join(buffer))
            buffer = []
            in_code = False
            continue

        buffer.append(line)

    if buffer:
        segments.append("".join(buffer))

    return segments


_SENTENCE_RE = re.compile(r".+?(?:[.!?。！？]+(?:\s+|$))", re.DOTALL)


def _split_text_segment_to_units(text: str) -> list[str]:
    parts = re.split(r"(\n{2,})", text)
    units: list[str] = []

    for part in parts:
        if not part:
            continue

        if re.fullmatch(r"\n{2,}", part):
            units.append(part)
            continue

        sentences = [m.group(0) for m in _SENTENCE_RE.finditer(part)]
        tail_start = sum(len(s) for s in sentences)
        tail = part[tail_start:]

        for s in sentences:
            if s:
                units.append(s)
        if tail.strip():
            units.append(tail)

    return units


def _finalize_chunks(chunks: list[list[str]]) -> list[str]:
    results: list[str] = []
    for units in chunks:
        merged = "".join(units).strip()
        if merged:
            results.append(merged)
    return results
