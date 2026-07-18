from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.dpo_data import parse_two_candidates, pick_preference
from app.finetune.eval_dataset import load_finetune_eval_cases


def test_parse_two_candidates_json() -> None:
    a, b = parse_two_candidates('{"a":"x","b":"y"}')
    assert a == "x"
    assert b == "y"


def test_pick_preference_uses_rubric_passed_first() -> None:
    case = load_finetune_eval_cases(Path("data/datasets/finetune_eval.sample.jsonl"))[0]
    chosen, rejected, meta = pick_preference(case=case, a="FastAPI @app.get /health", b="hello")
    assert chosen != rejected
    assert meta["reason"] in ("rubric", "heuristic_quality", "tie_breaker")

