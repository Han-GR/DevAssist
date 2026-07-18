from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.eval_dataset import load_finetune_eval_cases, summarize_eval_cases


def test_eval_dataset_sample_file_exists_and_has_enough_cases() -> None:
    path = Path("data/datasets/finetune_eval.sample.jsonl")
    cases = load_finetune_eval_cases(path)
    assert len(cases) >= 200


def test_eval_dataset_has_all_categories() -> None:
    path = Path("data/datasets/finetune_eval.sample.jsonl")
    cases = load_finetune_eval_cases(path)
    summary = summarize_eval_cases(cases)
    assert summary["normal"] > 0
    assert summary["edge"] > 0
    assert summary["adversarial"] > 0

