from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

import httpx

try:
    from app.rag.ingestion import ingest_text_document
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.rag.ingestion import ingest_text_document


GITHUB_TREE_URL = "https://api.github.com/repos/tiangolo/fastapi/git/trees/{ref}?recursive=1"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/tiangolo/fastapi/{ref}/{path}"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。

    Notes/Examples:
        最常用的用法：
        - docker compose up -d
        - docker compose run --rm backend python scripts/ingest_fastapi_docs.py
    """
    parser = argparse.ArgumentParser(description="Ingest FastAPI official docs (GitHub markdown) into Chroma.")
    parser.add_argument(
        "--ref",
        type=str,
        default="master",
        help="Git ref (default: master).",
    )
    parser.add_argument(
        "--docs-prefix",
        type=str,
        default="docs/en/docs/",
        help="Docs folder in repo (default: docs/en/docs/).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="fastapi_docs",
        help="Chroma collection name (default: fastapi_docs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of markdown files to ingest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matched markdown paths.",
    )
    return parser.parse_args()


async def fetch_markdown_paths(*, client: httpx.AsyncClient, ref: str, docs_prefix: str) -> list[str]:
    """
    从 GitHub tree API 拉取 FastAPI 仓库文件列表，并筛出文档 markdown 路径。

    Args:
        client (httpx.AsyncClient): HTTP 客户端。
        ref (str): Git ref。
        docs_prefix (str): 文档目录前缀。

    Returns:
        list[str]: markdown 路径列表（相对仓库根目录）。

    Raises:
        httpx.HTTPError: 请求失败时抛出。
        KeyError: 响应结构不符合预期时抛出。
    """
    url = GITHUB_TREE_URL.format(ref=ref)
    resp = await client.get(url, headers={"Accept": "application/vnd.github+json"}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    tree = data["tree"]

    paths: list[str] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = str(item.get("path", ""))
        if not path.startswith(docs_prefix):
            continue
        if not path.lower().endswith(".md"):
            continue
        paths.append(path)

    return sorted(paths)


async def fetch_markdown(*, client: httpx.AsyncClient, ref: str, path: str) -> str:
    """
    拉取一篇 markdown 文档的 raw 内容。

    Args:
        client (httpx.AsyncClient): HTTP 客户端。
        ref (str): Git ref。
        path (str): markdown 路径。

    Returns:
        str: 文档内容（UTF-8）。

    Raises:
        httpx.HTTPError: 请求失败时抛出。
    """
    url = GITHUB_RAW_URL.format(ref=ref, path=path)
    resp = await client.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.text


async def run() -> int:
    """
    入口：批量拉取 FastAPI 官方文档并写入知识库。

    Returns:
        int: 进程退出码，0 表示成功。

    Notes/Examples:
        - 该脚本会真实调用 embedding 服务，会产生 API 消耗；建议先用 --limit 小规模跑通。
        - 如果你只想看会 ingest 哪些文件，用 --dry-run。
    """
    args = parse_args()

    async with httpx.AsyncClient() as client:
        paths = await fetch_markdown_paths(client=client, ref=args.ref, docs_prefix=args.docs_prefix)
        if args.limit is not None:
            paths = paths[: args.limit]

        if not paths:
            print("no markdown files found")
            return 0

        print(f"matched {len(paths)} markdown files")
        if args.dry_run:
            for p in paths:
                print(p)
            return 0

        ok = 0
        failed = 0
        total_chunks = 0
        for path in paths:
            try:
                text = await fetch_markdown(client=client, ref=args.ref, path=path)
                document_id, chunk_count, collection = await ingest_text_document(
                    title=Path(path).name,
                    source=f"fastapi:{path}",
                    text=text,
                    collection_name=args.collection,
                )
            except Exception as exc:
                print(f"[fail] {path}: {exc}")
                failed += 1
                continue

            print(f"[ok] {path} -> {collection} chunks={chunk_count} document_id={document_id}")
            ok += 1
            total_chunks += chunk_count

        print(f"done: ok={ok} failed={failed} total_chunks={total_chunks}")
        return 0 if failed == 0 else 2


def main() -> None:
    """
    脚本入口。

    Notes/Examples:
        典型用法：
        - docker compose up -d
        - docker compose run --rm backend python scripts/ingest_fastapi_docs.py --limit 10
        - docker compose run --rm backend python scripts/ingest_fastapi_docs.py
    """
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
