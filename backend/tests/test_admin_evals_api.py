from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.admin_evals as admin_module


@dataclass
class _FakeEvalResult:
    id: UUID
    eval_type: str
    model_key: str
    metric_name: str
    scope: str
    score: float
    meta: dict[str, Any] | None
    created_at: str


def test_admin_eval_results_returns_list() -> None:
    client = TestClient(main_module.app)
    item_id = uuid4()

    async def _fake_list(
        *,
        limit: int = 200,
        eval_type: str | None = None,
        model_key: str | None = None,
        metric_name: str | None = None,
        scope: str | None = None,
    ):
        _ = (limit, eval_type, model_key, metric_name, scope)
        return [
            _FakeEvalResult(
                id=item_id,
                eval_type="finetune_rubric",
                model_key="base",
                metric_name="pass_rate",
                scope="all",
                score=0.5,
                meta={"evalset": "x"},
                created_at="2026-07-18T00:00:00+00:00",
            )
        ]

    admin_module.list_eval_results_from_db = _fake_list  # type: ignore[assignment]

    resp = client.get("/admin/eval-results?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert body[0]["id"] == str(item_id)
    assert body[0]["metric_name"] == "pass_rate"
    assert body[0]["score"] == 0.5


def test_admin_eval_result_detail_found() -> None:
    client = TestClient(main_module.app)
    item_id = uuid4()

    async def _fake_get(*, eval_id: UUID):  # noqa: ARG001
        from app.api.admin_evals import EvalResultItem  # noqa: PLC0415

        return EvalResultItem(
            id=item_id,
            eval_type="finetune_rubric",
            model_key="sft_lora",
            metric_name="avg_include_rate",
            scope="edge",
            score=0.8,
            meta=None,
            created_at="2026-07-18T00:00:00+00:00",
        )

    admin_module.get_eval_result_from_db = _fake_get  # type: ignore[assignment]

    resp = client.get(f"/admin/eval-results/{item_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(item_id)
    assert body["model_key"] == "sft_lora"
    assert body["scope"] == "edge"


def test_admin_eval_result_detail_not_found() -> None:
    client = TestClient(main_module.app)
    item_id = uuid4()

    async def _fake_get_none(*, eval_id: UUID):  # noqa: ARG001
        return None

    admin_module.get_eval_result_from_db = _fake_get_none  # type: ignore[assignment]

    resp = client.get(f"/admin/eval-results/{item_id}")
    assert resp.status_code == 404
