from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.judge import parse_judge_response, to_judge_result


def test_parse_judge_response_json_object() -> None:
    obj = parse_judge_response('{"score": 8, "passed": true, "reasons": ["ok"]}')
    assert obj["score"] == 8
    assert obj["passed"] is True
    assert obj["reasons"] == ["ok"]


def test_parse_judge_response_extract_braces() -> None:
    obj = parse_judge_response('noise {"score": 6.5, "passed": false, "reasons": ["bad"]} noise')
    assert obj["score"] == 6.5
    assert obj["passed"] is False


def test_parse_judge_response_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_judge_response("")


def test_parse_judge_response_non_object_raises() -> None:
    with pytest.raises(ValueError):
        parse_judge_response('["not", "object"]')


def test_to_judge_result_valid() -> None:
    r = to_judge_result({"score": 10, "passed": True, "reasons": ["great", " "]})
    assert r.score == 10.0
    assert r.passed is True
    assert r.reasons == ["great"]


def test_to_judge_result_invalid_score_range() -> None:
    with pytest.raises(ValueError):
        to_judge_result({"score": 11, "passed": True, "reasons": ["x"]})

