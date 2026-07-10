from __future__ import annotations

from app.rag.ingestion import _split_text_for_source


def test_split_text_for_code_uses_fixed_splitter() -> None:
    text = "\n".join([f"print({i})" for i in range(2000)])
    chunks = _split_text_for_source(text=text, source="main.py", chunk_size=512, overlap=64)
    assert len(chunks) > 1
    assert all(len(c) <= 512 for c in chunks)


def test_split_text_for_markdown_uses_semantic_splitter() -> None:
    text = "Hello world。\n\nThis is a test。\n\nAnd another paragraph。"
    chunks = _split_text_for_source(text=text, source="doc.md", chunk_size=32, overlap=0)
    assert len(chunks) >= 2
