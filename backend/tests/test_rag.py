from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.splitter import split_text
from app.rag.splitter import split_text_semantic


def test_split_text_empty_returns_empty_list() -> None:
    assert split_text("") == []


def test_split_text_short_text_returns_single_chunk() -> None:
    text = "hello"
    assert split_text(text, chunk_size=10, overlap=2) == ["hello"]


def test_split_text_creates_overlap() -> None:
    text = "".join(str(i % 10) for i in range(1200))
    chunks = split_text(text, chunk_size=512, overlap=64)

    assert len(chunks) >= 3
    assert chunks[0] == text[:512]
    assert chunks[1].startswith(text[512 - 64 : 512])
    assert "".join([chunks[0], chunks[1][64:]])[: (512 + 512 - 64)] == text[: (512 + 512 - 64)]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_split_text_invalid_params_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=chunk_size, overlap=overlap)


def test_split_text_semantic_keeps_fenced_code_block_intact() -> None:
    text = (
        "第一段。第二句。\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "结尾段落。"
    )

    chunks = split_text_semantic(text, chunk_size=40, overlap=0)
    merged = "\n".join(chunks)
    assert "```python" in merged
    assert "def add" in merged
    assert "```" in merged

    for c in chunks:
        if "```python" in c:
            assert c.strip().endswith("```")


def test_split_text_semantic_overlap_does_not_break_code_block_marker() -> None:
    text = (
        "开头段落。\n\n"
        "```txt\n"
        "line1\n"
        "line2\n"
        "```\n\n"
        "后续段落。"
    )
    chunks = split_text_semantic(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 2
    assert all("```txt" not in c or c.count("```") >= 2 for c in chunks)
