from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from app.rag.ingestion import ingest_text_document
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.rag.ingestion import ingest_text_document


def default_docs_dir() -> Path:
    """
    返回默认 docs 目录。

    Returns:
        Path: 默认文档目录路径。

    Notes/Examples:
        docker-compose 会把宿主机的 ./data 挂载到容器 /data，容器内默认用 /data/docs。
        本地直接跑脚本时，则回退到仓库内的 data/docs。
    """
    docker_path = Path("/data/docs")
    if docker_path.exists():
        return docker_path
    return REPO_ROOT / "data" / "docs"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。

    Notes/Examples:
        最常用的用法：
        - docker compose up -d
        - docker compose run --rm backend python scripts/ingest_docs.py
    """
    parser = argparse.ArgumentParser(description="Batch ingest documents from data/docs into Chroma.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(default_docs_dir()),
        help="Docs directory (default: /data/docs or data/docs).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".md,.txt,.py,.js",
        help="Comma-separated extensions (default: .md,.txt,.py,.js).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override collection name (default: CHROMA_COLLECTION).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of files to ingest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files, do not ingest.",
    )
    return parser.parse_args()


def iter_files(*, docs_dir: Path, extensions: set[str]) -> list[Path]:
    """
    扫描目录并返回匹配的文件列表。

    Args:
        docs_dir (Path): 文档目录。
        extensions (set[str]): 允许的扩展名集合（小写、带点）。

    Returns:
        list[Path]: 文件路径列表（按路径排序）。

    Raises:
        FileNotFoundError: docs_dir 不存在时抛出。
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"docs dir not found: {docs_dir}")

    files: list[Path] = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    return sorted(files)


async def run() -> int:
    """
    批量 ingestion 主流程。

    Returns:
        int: 进程退出码，0 表示成功。

    Notes/Examples:
        - 每个文件会写入一条 documents 记录，并写入向量库。
        - 如果某个文件失败，会打印错误并继续处理下一个文件。
    """
    args = parse_args()
    docs_dir = Path(args.path).resolve()
    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    files = iter_files(docs_dir=docs_dir, extensions=exts)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print(f"no files found in {docs_dir} with extensions: {sorted(exts)}")
        return 0

    print(f"found {len(files)} files under {docs_dir}")
    if args.dry_run:
        for p in files:
            print(p)
        return 0

    ok = 0
    failed = 0
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[skip] {p}: not utf-8")
            failed += 1
            continue

        try:
            source = str(p.relative_to(docs_dir))
        except ValueError:
            source = str(p)
        try:
            document_id, chunk_count, collection = await ingest_text_document(
                title=p.name,
                source=source,
                text=text,
                collection_name=args.collection,
            )
        except Exception as exc:
            print(f"[fail] {p}: {exc}")
            failed += 1
            continue

        print(f"[ok] {p} -> {collection} chunks={chunk_count} document_id={document_id}")
        ok += 1

    print(f"done: ok={ok} failed={failed}")
    return 0 if failed == 0 else 2


def main() -> None:
    """
    脚本入口。

    Notes/Examples:
        该脚本会使用 backend/.env 的配置（embedding、chroma、db），确保服务已启动：
        - docker compose up -d
        - docker compose run --rm backend python scripts/ingest_docs.py
    """
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
