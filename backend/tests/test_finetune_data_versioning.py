from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.data_versioning import build_dataset_fingerprint, create_dataset_snapshot


def test_build_dataset_fingerprint(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    (repo_root / ".git" / "HEAD").write_text("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n", encoding="utf-8")

    f = repo_root / "data" / "datasets" / "a.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")

    fp = build_dataset_fingerprint(file_path=f, root_dir=repo_root)
    assert fp.relative_path == "data/datasets/a.jsonl"
    assert fp.non_empty_lines == 2
    assert len(fp.sha256) == 64


def test_create_dataset_snapshot(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (repo_root / ".git" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (repo_root / ".git" / "refs" / "heads" / "main").write_text(
        "0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8"
    )

    f = repo_root / "data" / "datasets" / "a.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"a":1}\n', encoding="utf-8")

    out = create_dataset_snapshot(
        input_files=[f],
        repo_root=repo_root,
        snapshot_root=tmp_path / "snapshots",
        label="t",
        snapshot_id="s1",
    )
    assert (out / "manifest.json").exists()
    assert (out / "data" / "datasets" / "a.jsonl").exists()


def test_create_dataset_snapshot_empty_inputs_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_dataset_snapshot(
            input_files=[],
            repo_root=tmp_path,
            snapshot_root=tmp_path / "snapshots",
        )

