from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelVersionSpec:
    base_model: str
    variant: str
    date: str
    commit: str


def _slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"[^a-z0-9.\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def normalize_base_model_name(model_name_or_path: str) -> str:
    """
    将 base model 名称规范化为用于目录名的短字符串。

    Args:
        model_name_or_path: HuggingFace 模型名或本地路径名（例如 "Qwen/Qwen2.5-7B-Instruct"）。

    Returns:
        适合用作目录名的字符串（例如 "qwen2.5-7b"）。

    Raises:
        ValueError: model_name_or_path 为空。
    """

    raw = (model_name_or_path or "").strip()
    if not raw:
        raise ValueError("model_name_or_path must not be empty")

    last = raw.split("/")[-1]
    s = _slugify(last)
    for suffix in ("-instruct", "-chat", "-base"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s or _slugify(raw)


def detect_git_commit_short(*, repo_root: Path) -> str | None:
    """
    读取 .git 信息推断当前 commit（不依赖 git 命令）。

    Args:
        repo_root: 仓库根目录（包含 .git 的目录）。

    Returns:
        7 位短 commit；如果无法推断则返回 None。
    """

    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None

    head_text = head.read_text(encoding="utf-8").strip()
    if not head_text:
        return None

    if head_text.startswith("ref: "):
        ref = head_text.replace("ref: ", "", 1).strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            sha = ref_path.read_text(encoding="utf-8").strip()
            return sha[:7] if sha else None

        packed = repo_root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                l = line.strip()
                if not l or l.startswith("#") or l.startswith("^"):
                    continue
                parts = l.split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0][:7] if parts[0] else None
        return None

    return head_text[:7]


def build_model_version(*, base_model: str, variant: str, date: str | None = None, commit: str | None = None) -> str:
    """
    构造微调模型版本字符串。

    Args:
        base_model: base model 名称（例如 "Qwen/Qwen2.5-7B-Instruct"）。
        variant: 版本类型（例如 "lora" / "dpo-lora"）。
        date: 可选，日期字符串（YYYYMMDD）。不传则使用本地当前日期。
        commit: 可选，git commit 短 hash（7 位）。不传则使用 "unknown"。

    Returns:
        版本字符串，格式为 "{base}-{variant}-{date}-{commit}"。

    Raises:
        ValueError: base_model 或 variant 为空。
    """

    v = (variant or "").strip()
    if not v:
        raise ValueError("variant must not be empty")

    base = normalize_base_model_name(base_model)
    date_str = (date or "").strip() or datetime.now().strftime("%Y%m%d")
    commit_str = (commit or "").strip() or "unknown"
    return f"{base}-{_slugify(v)}-{_slugify(date_str)}-{_slugify(commit_str)}"


def build_model_version_spec(*, base_model: str, variant: str, repo_root: Path, date: str | None = None) -> ModelVersionSpec:
    """
    一次性构造版本所需的全部字段，方便落库或写入日志。

    Args:
        base_model: base model 名称或路径。
        variant: 版本类型。
        repo_root: git 仓库根目录。
        date: 可选日期（YYYYMMDD）。

    Returns:
        ModelVersionSpec。
    """

    date_str = (date or "").strip() or datetime.now().strftime("%Y%m%d")
    commit = detect_git_commit_short(repo_root=repo_root) or "unknown"
    return ModelVersionSpec(
        base_model=normalize_base_model_name(base_model),
        variant=_slugify(variant),
        date=_slugify(date_str),
        commit=_slugify(commit),
    )

