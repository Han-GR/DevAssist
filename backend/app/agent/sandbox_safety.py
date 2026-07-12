"""
沙箱安全控制模块。

职责：
1. 危险操作检测：扫描代码中是否包含高风险调用（文件写入、网络请求、系统命令等）。
2. 文件路径白名单校验：若代码中出现文件路径，确保其在允许的目录范围内。

设计原则：
- 这里只做"静态文本扫描"，不做运行时拦截（运行时隔离由 Docker 沙箱负责）。
- 检测结果分两级：WARNING（可继续执行，但需告知调用方）和 BLOCKED（直接拒绝执行）。
- 路径白名单默认为空列表，表示"不限制路径"；非空时才做校验。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 危险模式定义
# ---------------------------------------------------------------------------

# 每条规则：(pattern, level, reason)
# level: "blocked" | "warning"
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # 文件写入
    (re.compile(r"\bopen\s*\(.*['\"]w['\"]", re.IGNORECASE), "blocked", "file write via open()"),
    (re.compile(r"\bopen\s*\(.*['\"]a['\"]", re.IGNORECASE), "blocked", "file append via open()"),
    (re.compile(r"\bopen\s*\(.*['\"]x['\"]", re.IGNORECASE), "blocked", "file create via open()"),
    (re.compile(r"\bshutil\.(copy|move|rmtree|copytree)\b"), "blocked", "file system operation via shutil"),
    (re.compile(r"\bos\.(remove|unlink|rmdir|makedirs|mkdir|rename|replace)\b"), "blocked", "file system operation via os"),
    (re.compile(r"\bpathlib\.Path.*\.write_"), "blocked", "file write via pathlib"),
    # 网络请求
    (re.compile(r"\b(requests|httpx|urllib|aiohttp|socket)\b"), "blocked", "network access"),
    (re.compile(r"\bsubprocess\b"), "blocked", "subprocess execution"),
    # 系统命令
    (re.compile(r"\bos\.(system|popen|execv|execve|execvp|spawnl|spawnle)\b"), "blocked", "system command via os"),
    (re.compile(r"\b__import__\s*\("), "warning", "dynamic import via __import__"),
    (re.compile(r"\bimportlib\b"), "warning", "dynamic import via importlib"),
    # 危险内建
    (re.compile(r"\beval\s*\("), "warning", "eval() usage"),
    (re.compile(r"\bexec\s*\("), "warning", "exec() usage"),
    (re.compile(r"\bcompile\s*\("), "warning", "compile() usage"),
    # 进程/信号
    (re.compile(r"\bos\.(kill|getpid|getppid)\b"), "warning", "process control via os"),
    (re.compile(r"\bsignal\b"), "warning", "signal module usage"),
]

# 提取代码中出现的文件路径（简单启发式：字符串字面量中含 / 或 \ 的）
_PATH_LITERAL_RE = re.compile(r"""['"]((?:[A-Za-z]:)?[/\\][^'"]{1,300})['"]""")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SafetyIssue:
    """
    单条安全问题记录。

    Args:
        level (str): "blocked" 或 "warning"。
        reason (str): 问题描述。
        line_no (int | None): 出现的行号（1-based），None 表示未定位。
    """
    level: str
    reason: str
    line_no: int | None = None


@dataclass
class SafetyCheckResult:
    """
    安全检查结果。

    Args:
        is_blocked (bool): 是否包含 blocked 级别问题（True 时应拒绝执行）。
        issues (list[SafetyIssue]): 所有检测到的问题列表。
    """
    is_blocked: bool
    issues: list[SafetyIssue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 核心检查函数
# ---------------------------------------------------------------------------

def check_code_safety(
    *,
    code: str,
    allowed_paths: list[str] | None = None,
) -> SafetyCheckResult:
    """
    对代码字符串做静态安全扫描。

    Args:
        code (str): 待检查的 Python 代码。
        allowed_paths (list[str] | None): 允许访问的文件路径前缀列表；
            None 或空列表表示不做路径限制。

    Returns:
        SafetyCheckResult: 检查结果，包含 is_blocked 与 issues 列表。

    Raises:
        None: 所有异常均被捕获，不影响调用方。

    Notes/Examples:
        - 只做静态文本扫描，不执行代码。
        - 路径校验基于字符串字面量启发式提取，存在误报/漏报，
          真正的隔离依赖 Docker 沙箱的 read_only + tmpfs 配置。
    """
    issues: list[SafetyIssue] = []
    lines = code.splitlines()

    # 1. 危险模式扫描（逐行）
    for line_no, line in enumerate(lines, start=1):
        for pattern, level, reason in _DANGEROUS_PATTERNS:
            if pattern.search(line):
                issues.append(SafetyIssue(level=level, reason=reason, line_no=line_no))

    # 2. 文件路径白名单校验（仅当 allowed_paths 非空时生效）
    if allowed_paths:
        resolved_allowed = [str(Path(p).resolve()) for p in allowed_paths]
        for line_no, line in enumerate(lines, start=1):
            for path_str in _PATH_LITERAL_RE.findall(line):
                try:
                    resolved = str(Path(path_str).resolve())
                except Exception:
                    continue
                if not any(resolved.startswith(a) for a in resolved_allowed):
                    issues.append(
                        SafetyIssue(
                            level="blocked",
                            reason=f"path not in allowlist: {path_str!r}",
                            line_no=line_no,
                        )
                    )

    is_blocked = any(i.level == "blocked" for i in issues)
    return SafetyCheckResult(is_blocked=is_blocked, issues=issues)


def format_safety_report(result: SafetyCheckResult) -> str:
    """
    将安全检查结果格式化为可读文本，供日志或错误消息使用。

    Args:
        result (SafetyCheckResult): 安全检查结果。

    Returns:
        str: 格式化后的报告文本。

    Raises:
        None
    """
    if not result.issues:
        return "No safety issues detected."
    lines = [f"Safety check {'BLOCKED' if result.is_blocked else 'WARNING'}:"]
    for issue in result.issues:
        loc = f" (line {issue.line_no})" if issue.line_no is not None else ""
        lines.append(f"  [{issue.level.upper()}]{loc} {issue.reason}")
    return "\n".join(lines)
