from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.splitter import split_text


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

