from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.eval_dataset import load_finetune_eval_cases
from app.finetune.eval_runner import aggregate_rubric_results, evaluate_with_rubric


def test_rubric_eval_and_aggregation_end_to_end(tmp_path: Path) -> None:
    evalset = tmp_path / "finetune_eval.jsonl"
    evalset.write_text(
        "\n".join(
            [
                (
                    '{"id":"e1","category":"normal","instruction":"i","input":"q","rubric":'
                    '{"must_include":["FastAPI","Pydantic"],"must_not_include":["sk-"]}}'
                ),
                (
                    '{"id":"e2","category":"edge","instruction":"i","input":"q","rubric":'
                    '{"must_include":["async"],"must_not_include":[]}}'
                ),
                (
                    '{"id":"e3","category":"adversarial","instruction":"i","input":"q","rubric":'
                    '{"must_include":["docker"],"must_not_include":["AKIA"]}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_finetune_eval_cases(evalset)
    assert [c.id for c in cases] == ["e1", "e2", "e3"]

    answers = {
        "e1": "FastAPI uses Pydantic to validate data.",
        "e2": "Use sync code here.",
        "e3": "Run in Docker. Do not leak AKIA keys.",
    }

    rows: list[dict[str, object]] = []
    for c in cases:
        rr = evaluate_with_rubric(case=c, answer=answers[c.id])
        rows.append(
            {
                "id": c.id,
                "category": c.category,
                "include_rate": rr.include_rate,
                "passed": rr.passed,
                "violated_count": len(rr.violated_terms),
            }
        )

    summary_all = aggregate_rubric_results(rows)
    assert summary_all["count"] == 3
    assert summary_all["pass_rate"] == 1 / 3
    assert summary_all["violation_rate"] == 1 / 3
    assert abs(float(summary_all["avg_include_rate"]) - (2 / 3)) < 1e-9

    summary_normal = aggregate_rubric_results([r for r in rows if r["category"] == "normal"])
    assert summary_normal["count"] == 1
    assert summary_normal["pass_rate"] == 1.0
    assert summary_normal["violation_rate"] == 0.0
    assert summary_normal["avg_include_rate"] == 1.0


def test_rubric_eval_requires_non_empty_answer(tmp_path: Path) -> None:
    evalset = tmp_path / "finetune_eval.jsonl"
    evalset.write_text(
        '{"id":"e1","category":"normal","instruction":"i","input":"q","rubric":{"must_include":[],"must_not_include":[]}}\n',
        encoding="utf-8",
    )
    case = load_finetune_eval_cases(evalset)[0]
    try:
        evaluate_with_rubric(case=case, answer=" ")
    except ValueError as e:
        assert "answer" in str(e)
    else:
        raise AssertionError("expected ValueError")

