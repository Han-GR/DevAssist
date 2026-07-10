from __future__ import annotations


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
