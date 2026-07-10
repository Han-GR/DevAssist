from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from uuid import UUID

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main as main_module
import app.api.ingest as ingest_module


@dataclass
class _FakeCollection:
    last_add: dict[str, object] | None = None

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        self.last_add = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }


class _FakeChromaManager:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(self, name: str):
        return self.collection


class _FakeEmbedder:
    async def embed_texts(self, texts: list[str], *, batch_size: int = 96):
        return [[1.0] for _ in texts]


def test_ingest_rejects_unsupported_file_type() -> None:
    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_ingest_stores_chunks_in_chroma(monkeypatch) -> None:
    fake_mgr = _FakeChromaManager()
    monkeypatch.setattr(ingest_module, "embedder", _FakeEmbedder())
    monkeypatch.setattr(ingest_module, "chroma_manager", fake_mgr)
    async def _fake_persist_document_to_db(**kwargs):
        return UUID("00000000-0000-0000-0000-000000000001")

    monkeypatch.setattr(ingest_module, "persist_document_to_db", _fake_persist_document_to_db)

    client = TestClient(main_module.app)
    resp = client.post(
        "/ingest",
        files={"file": ("doc.md", "hello\n\nworld".encode("utf-8"), "text/markdown")},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["document_id"] == "00000000-0000-0000-0000-000000000001"
    assert body["filename"] == "doc.md"
    assert body["chunk_count"] >= 1

    assert fake_mgr.collection.last_add
    last_add = fake_mgr.collection.last_add
    assert len(last_add["ids"]) == len(last_add["documents"]) == len(last_add["embeddings"])
