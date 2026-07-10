from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.chroma import ChromaCollectionManager


@dataclass
class _FakeCollection:
    name: str


class _FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, *, name: str):
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name=name)
        return self.collections[name]


def test_get_or_create_collection_creates_once() -> None:
    client = _FakeClient()
    mgr = ChromaCollectionManager(host="x", port=8000, client=client)

    a = mgr.get_or_create_collection(name="devassist")
    b = mgr.get_or_create_collection(name="devassist")

    assert a is b
    assert a.name == "devassist"


def test_get_or_create_collection_empty_name_raises() -> None:
    client = _FakeClient()
    mgr = ChromaCollectionManager(host="x", port=8000, client=client)
    with pytest.raises(ValueError):
        mgr.get_or_create_collection(name="  ")


def test_invalid_port_raises() -> None:
    with pytest.raises(ValueError):
        ChromaCollectionManager(host="x", port=0)

