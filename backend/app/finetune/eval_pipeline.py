from __future__ import annotations

from dataclasses import dataclass
import io
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

from contextlib import redirect_stdout


@dataclass(frozen=True)
class PipelineStepResult:
    name: str
    status: str
    outputs: dict[str, str]
    summary: dict[str, Any] | None = None
    error: str | None = None


def _run_with_argv(*, argv: list[str], fn: Callable[[], int]) -> tuple[int, str]:
    old_argv = sys.argv[:]
    buf = io.StringIO()
    try:
        sys.argv = argv[:]
        with redirect_stdout(buf):
            code = int(fn())
        return code, buf.getvalue()
    finally:
        sys.argv = old_argv


def _parse_first_json_line(text: str) -> dict[str, Any] | None:
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_key_value_lines(text: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in (text or "").splitlines():
        raw = line.strip()
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            kv[k] = v
    return kv


def _write_markdown(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _load_script_module(*, script_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts"


def run_finetune_eval_pipeline(

    *,
    evalset: Path,
    base_model: str,
    dpo_adapter: str | None,
    limit: int | None,
    out_dir: Path,
    enable_rubric: bool = True,
    enable_judge: bool = False,
    judge_provider: str = "deepseek",
    judge_model: str | None = None,
) -> tuple[list[PipelineStepResult], Path]:
    """
    运行微调评测流水线并生成总报告（best-effort）。

    Args:
        evalset: finetune_eval 数据集路径。
        base_model: base 模型名或路径。
        sft_adapter: SFT LoRA adapter 路径（可选）。
        dpo_adapter: DPO LoRA adapter 路径（可选）。
        limit: 限制评测条数（可选）。
        out_dir: 输出目录（用于写入各子报告与总报告）。
        enable_rubric: 是否启用 rubric 评测（base vs sft、three-way）。
        enable_judge: 是否启用 LLM-as-Judge 评测（需要 LLM 配置与依赖）。
        judge_provider: judge 的 provider（传给 LLMClient）。
        judge_model: 可选，覆盖 judge 的 model。

    Returns:
        (steps, report_path)。
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[PipelineStepResult] = []

    if enable_rubric:
        steps.append(
            _run_eval_sft_vs_base(
                evalset=evalset,
                base_model=base_model,
                adapter=sft_adapter,
                limit=limit,
                out_md=out_dir / "finetune_sft_vs_base_report.md",
            )
        )
        steps.append(
            _run_eval_three_way(
                evalset=evalset,
                base_model=base_model,
                sft_adapter=sft_adapter,
                dpo_adapter=dpo_adapter,
                limit=limit,
                out_md=out_dir / "finetune_three_way_report.md",
            )
        )

    if enable_judge:
        steps.append(
            _run_judge_eval(
                target="base",
                evalset=evalset,
                base_model=base_model,
                adapter=None,
                limit=limit,
                out_json=out_dir / "judge_report.base.json",
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
        )
        if sft_adapter:
            steps.append(
                _run_judge_eval(
                    target="sft_lora",
                    evalset=evalset,
                    base_model=base_model,
                    adapter=sft_adapter,
                    limit=limit,
                    out_json=out_dir / "judge_report.lora.json",
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
            )

    report_path = out_dir / "finetune_eval_pipeline_report.md"
    _write_markdown(path=report_path, content=_render_pipeline_report(steps=steps, evalset=evalset))
    return steps, report_path


def _render_pipeline_report(*, steps: list[PipelineStepResult], evalset: Path) -> str:
    lines: list[str] = []
    lines.append("# Fine-tune Evaluation Pipeline Report")
    lines.append("")
    lines.append(f"- evalset: `{evalset}`")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append("| step | status | outputs |")
    lines.append("|---|---|---|")
    for s in steps:
        outputs = ", ".join([f"`{k}`={v}" for k, v in s.outputs.items()]) if s.outputs else "(none)"
        lines.append(f"| {s.name} | {s.status} | {outputs} |")
    lines.append("")

    failures = [s for s in steps if s.status not in ("ok", "skipped")]
    if failures:
        lines.append("## Errors")
        lines.append("")
        for s in failures:
            if s.error:
                lines.append(f"### {s.name}")
                lines.append("")
                lines.append("```")
                lines.append(s.error.rstrip())
                lines.append("```")
                lines.append("")
    return "\n".join(lines)


def _run_eval_sft_vs_base(*, evalset: Path, base_model: str, adapter: str | None, limit: int | None, out_md: Path) -> PipelineStepResult:
    try:
        m = _load_script_module(
            script_path=_scripts_dir() / "eval_sft_vs_base.py",
            module_name="eval_sft_vs_base",
        )
    except Exception as exc:
        return PipelineStepResult(
            name="rubric_sft_vs_base",
            status="skipped",
            outputs={},
            error=f"import failed: {exc}",
        )

    argv: list[str] = [
        "eval_sft_vs_base.py",
        "--evalset",
        str(evalset),
        "--base-model",
        str(base_model),
        "--output-md",
        str(out_md),
    ]
    if adapter:
        argv += ["--adapter", str(adapter)]
    if limit is not None:
        argv += ["--limit", str(int(limit))]

    try:
        code, out = _run_with_argv(argv=argv, fn=m.main)
        kv = _parse_key_value_lines(out)
        summary = _parse_first_json_line(out)
        status = "ok" if code == 0 else "failed"
        outputs = {"report_md": kv.get("report_md", str(out_md))}
        return PipelineStepResult(name="rubric_sft_vs_base", status=status, outputs=outputs, summary=summary)
    except Exception as exc:
        return PipelineStepResult(name="rubric_sft_vs_base", status="failed", outputs={"report_md": str(out_md)}, error=str(exc))


def _run_eval_three_way(
    *,
    evalset: Path,
    base_model: str,
    sft_adapter: str | None,
    dpo_adapter: str | None,
    limit: int | None,
    out_md: Path,
) -> PipelineStepResult:
    try:
        m = _load_script_module(
            script_path=_scripts_dir() / "eval_base_sft_dpo.py",
            module_name="eval_base_sft_dpo",
        )
    except Exception as exc:
        return PipelineStepResult(
            name="rubric_three_way",
            status="skipped",
            outputs={},
            error=f"import failed: {exc}",
        )

    argv: list[str] = [
        "eval_base_sft_dpo.py",
        "--evalset",
        str(evalset),
        "--base-model",
        str(base_model),
        "--output-md",
        str(out_md),
    ]
    if sft_adapter:
        argv += ["--sft-adapter", str(sft_adapter)]
    if dpo_adapter:
        argv += ["--dpo-adapter", str(dpo_adapter)]
    if limit is not None:
        argv += ["--limit", str(int(limit))]

    try:
        code, out = _run_with_argv(argv=argv, fn=m.main)
        kv = _parse_key_value_lines(out)
        summary = _parse_first_json_line(out)
        status = "ok" if code == 0 else "failed"
        outputs = {"report_md": kv.get("report_md", str(out_md))}
        return PipelineStepResult(name="rubric_three_way", status=status, outputs=outputs, summary=summary)
    except Exception as exc:
        return PipelineStepResult(name="rubric_three_way", status="failed", outputs={"report_md": str(out_md)}, error=str(exc))


def _run_judge_eval(
    *,
    target: str,
    evalset: Path,
    base_model: str,
    adapter: str | None,
    limit: int | None,
    out_json: Path,
    judge_provider: str,
    judge_model: str | None,
) -> PipelineStepResult:
    try:
        m = _load_script_module(
            script_path=_scripts_dir() / "judge_eval.py",
            module_name="judge_eval",
        )
    except Exception as exc:
        return PipelineStepResult(
            name=f"judge_{target}",
            status="skipped",
            outputs={},
            error=f"import failed: {exc}",
        )

    argv: list[str] = [
        "judge_eval.py",
        "--evalset",
        str(evalset),
        "--base-model",
        str(base_model),
        "--output-json",
        str(out_json),
        "--judge-provider",
        str(judge_provider),
    ]
    if adapter:
        argv += ["--adapter", str(adapter)]
    if judge_model:
        argv += ["--judge-model", str(judge_model)]
    if limit is not None:
        argv += ["--limit", str(int(limit))]

    try:
        import asyncio

        def _main() -> int:
            return int(asyncio.run(m.main_async()))

        code, out = _run_with_argv(argv=argv, fn=_main)
        kv = _parse_key_value_lines(out)
        summary = _parse_first_json_line(out)
        status = "ok" if code == 0 else "failed"
        outputs = {"output_json": kv.get("output_json", str(out_json))}
        return PipelineStepResult(name=f"judge_{target}", status=status, outputs=outputs, summary=summary)
    except Exception as exc:
        return PipelineStepResult(name=f"judge_{target}", status="failed", outputs={"output_json": str(out_json)}, error=str(exc))
