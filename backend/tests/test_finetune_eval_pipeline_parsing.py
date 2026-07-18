from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.eval_pipeline import _parse_first_json_line, _parse_key_value_lines


def test_parse_first_json_line() -> None:
    text = "noise\n{\"a\":1}\nmore\n"
    assert _parse_first_json_line(text) == {"a": 1}


def test_parse_key_value_lines() -> None:
    text = "x=1\ny = 2\nnotkv\n"
    assert _parse_key_value_lines(text) == {"x": "1", "y": "2"}

