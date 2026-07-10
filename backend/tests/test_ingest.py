from __future__ import annotations

from pathlib import Path
import sys
from uuid import UUID

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.ingest as ingest_module


def test_ingest_rejects_unsupported_file_type() -> None:
    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_ingest_stores_chunks_in_chroma(monkeypatch) -> None:
    async def _fake_ingest_text_document(**kwargs):
        return UUID("00000000-0000-0000-0000-000000000001"), 2, "devassist"

    monkeypatch.setattr(ingest_module, "ingest_text_document", _fake_ingest_text_document)

    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("doc.md", "hello\n\nworld".encode("utf-8"), "text/markdown")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["document_id"] == "00000000-0000-0000-0000-000000000001"
    assert body["filename"] == "doc.md"
    assert body["chunk_count"] == 2
    assert body["collection"] == "devassist"


def test_ingest_accepts_code_files(monkeypatch) -> None:
    async def _fake_ingest_text_document(**kwargs):
        return UUID("00000000-0000-0000-0000-000000000001"), 1, "devassist"

    monkeypatch.setattr(ingest_module, "ingest_text_document", _fake_ingest_text_document)

    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("main.py", "print('hi')".encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200
