from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from app.finetune.versioning import detect_git_commit_short


@dataclass(frozen=True)
class DatasetFileFingerprint:
    relative_path: str
    sha256: str
    size_bytes: int
    non_empty_lines: int


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    created_at: str
    commit: str
    label: str | None
    files: list[DatasetFileFingerprint]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_non_empty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_dataset_fingerprint(*, file_path: Path, root_dir: Path) -> DatasetFileFingerprint:
    """
    为单个数据文件构造指纹信息。

    Args:
        file_path: 数据文件路径。
        root_dir: 根目录，用于生成 relative_path（便于跨机器复现）。

    Returns:
        DatasetFileFingerprint。

    Raises:
        FileNotFoundError: file_path 不存在。
        ValueError: file_path 不在 root_dir 下。
        OSError: 读文件失败。
    """

    p = file_path.resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    root = root_dir.resolve()
    try:
        rel = p.relative_to(root)
    except Exception as exc:
        raise ValueError(f"file_path must be under root_dir: {p}") from exc

    stat = p.stat()
    return DatasetFileFingerprint(
        relative_path=str(rel).replace("\\", "/"),
        sha256=_sha256_file(p),
        size_bytes=int(stat.st_size),
        non_empty_lines=_count_non_empty_lines(p),
    )


def create_dataset_snapshot(
    *,
    input_files: list[Path],
    repo_root: Path,
    snapshot_root: Path,
    label: str | None = None,
    snapshot_id: str | None = None,
) -> Path:
    """
    创建数据集快照（复制文件 + 写入 manifest.json）。

    Args:
        input_files: 需要纳入快照的数据文件列表（建议为 JSONL）。
        repo_root: 仓库根目录（用于推断 commit + 生成 relative_path）。
        snapshot_root: 快照根目录（例如 data/datasets/snapshots）。
        label: 可选标签（例如 "pre-clean" / "sft-500"）。
        snapshot_id: 可选，覆盖默认快照 ID（默认 {YYYYMMDD}-{commit}）。

    Returns:
        本次快照目录路径。

    Raises:
        ValueError: input_files 为空或包含重复文件。
        FileNotFoundError: 任一输入文件不存在。
        OSError: 复制或写文件失败。
    """

    if not input_files:
        raise ValueError("input_files must not be empty")

    resolved: list[Path] = [p.resolve() for p in input_files]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input_files must not contain duplicates")

    commit = detect_git_commit_short(repo_root=repo_root.resolve()) or "unknown"
    date_str = datetime.now().strftime("%Y%m%d")
    sid = (snapshot_id or "").strip() or f"{date_str}-{commit}"

    out_dir = snapshot_root.resolve() / sid
    out_dir.mkdir(parents=True, exist_ok=False)

    fingerprints: list[DatasetFileFingerprint] = []
    for p in resolved:
        fp = build_dataset_fingerprint(file_path=p, root_dir=repo_root)
        fingerprints.append(fp)

        src = (repo_root.resolve() / fp.relative_path).resolve()
        dst = out_dir / fp.relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    snapshot = DatasetSnapshot(
        snapshot_id=sid,
        created_at=datetime.now().isoformat(timespec="seconds"),
        commit=commit,
        label=(label or "").strip() or None,
        files=fingerprints,
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_to_json(snapshot), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def _to_json(snapshot: DatasetSnapshot) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "created_at": snapshot.created_at,
        "commit": snapshot.commit,
        "label": snapshot.label,
        "files": [
            {
                "relative_path": f.relative_path,
                "sha256": f.sha256,
                "size_bytes": f.size_bytes,
                "non_empty_lines": f.non_empty_lines,
            }
            for f in snapshot.files
        ],
    }

