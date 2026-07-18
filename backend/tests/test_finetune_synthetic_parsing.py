from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.synthetic import parse_sft_items_from_text


def test_parse_sft_items_from_text_json_array() -> None:
    text = '[{"input":"q1","output":"a1"},{"input":"q2","output":"a2"}]'
    items = parse_sft_items_from_text(text)
    assert items == [{"input": "q1", "output": "a1"}, {"input": "q2", "output": "a2"}]


def test_parse_sft_items_from_text_jsonl() -> None:
    text = '{"input":"q1","output":"a1"}\n{"input":"q2","output":"a2"}\n'
    items = parse_sft_items_from_text(text)
    assert items == [{"input": "q1", "output": "a1"}, {"input": "q2", "output": "a2"}]


def test_parse_sft_items_from_text_bracketed_substring() -> None:
    text = 'noise\\n[{"input":"q1","output":"a1"}]\\nnoise'
    items = parse_sft_items_from_text(text)
    assert items == [{"input": "q1", "output": "a1"}]

