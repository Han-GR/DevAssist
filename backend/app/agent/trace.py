from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog


@dataclass(frozen=True)
class TraceStep:
    """
    Agent 单步 Trace 记录。

    Args:
        step_index (int): 第几步（从 0 开始）。
        thought (str): 模型输出的 Thought（可能为空）。
        action_raw (str): 模型输出的 Action 原文。
        tool_name (str | None): 若调用工具，则为工具名；否则为 None。
        tool_args (dict[str, Any] | None): 工具入参；非工具调用为 None。
        observation (Any | None): 工具输出；非工具调用为 None。
        error (str | None): 本步错误信息（如解析失败/工具异常），无则为 None。
        started_at_ms (int): 本步开始时间戳（毫秒）。
        finished_at_ms (int): 本步结束时间戳（毫秒）。

    Returns:
        TraceStep: 记录对象。

    Raises:
        None
    """

    step_index: int
    thought: str
    action_raw: str
    tool_name: str | None
    tool_args: dict[str, Any] | None
    observation: Any | None
    error: str | None
    started_at_ms: int
    finished_at_ms: int

    @property
    def latency_ms(self) -> int:
        """
        计算本步耗时（毫秒）。

        Args:
            None

        Returns:
            int: 耗时（毫秒）。

        Raises:
            None
        """
        return max(0, self.finished_at_ms - self.started_at_ms)

    def to_dict(self) -> dict[str, Any]:
        """
        导出为可序列化的 dict，便于后续落库或前端展示。

        Args:
            None

        Returns:
            dict[str, Any]: JSON 友好的结构。

        Raises:
            None
        """
        return {
            "step_index": self.step_index,
            "thought": self.thought,
            "action_raw": self.action_raw,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "observation": self.observation,
            "error": self.error,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "latency_ms": self.latency_ms,
        }


class TraceRecorder:
    """
    Trace 记录器：收集每一步的 Thought/Action/Observation，并输出结构化日志。

    Args:
        run_id (str | None): 可选的运行标识，用于把多条日志关联起来。

    Returns:
        TraceRecorder: 记录器实例。

    Raises:
        None
    """

    def __init__(self, *, run_id: str | None = None) -> None:
        self._run_id = run_id
        self._steps: list[TraceStep] = []
        self._logger = structlog.get_logger()

    def start_step(self, *, step_index: int) -> int:
        """
        记录本步开始时间。

        Args:
            step_index (int): 当前步序号。

        Returns:
            int: started_at_ms（毫秒时间戳）。

        Raises:
            None
        """
        _ = step_index
        return int(time.time() * 1000)

    def finish_step(
        self,
        *,
        step_index: int,
        started_at_ms: int,
        thought: str,
        action_raw: str,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        observation: Any | None,
        error: str | None,
    ) -> TraceStep:
        """
        结束本步并写入记录，同时输出结构化日志。

        Args:
            step_index (int): 当前步序号。
            started_at_ms (int): 本步开始时间戳（毫秒）。
            thought (str): Thought 文本。
            action_raw (str): Action 原文。
            tool_name (str | None): 工具名（如有）。
            tool_args (dict[str, Any] | None): 工具入参（如有）。
            observation (Any | None): 工具输出（如有）。
            error (str | None): 错误信息（如有）。

        Returns:
            TraceStep: 生成的单步记录对象。

        Raises:
            None
        """
        finished_at_ms = int(time.time() * 1000)
        step = TraceStep(
            step_index=step_index,
            thought=thought,
            action_raw=action_raw,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
            error=error,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
        )
        self._steps.append(step)

        self._logger.info(
            "agent_trace_step",
            run_id=self._run_id,
            step_index=step_index,
            tool_name=tool_name,
            latency_ms=step.latency_ms,
            success=error is None,
        )
        return step

    def steps(self) -> list[TraceStep]:
        """
        获取所有 trace steps（按顺序）。

        Args:
            None

        Returns:
            list[TraceStep]: steps 列表。

        Raises:
            None
        """
        return list(self._steps)

    def to_dict(self) -> dict[str, Any]:
        """
        导出整次运行的 trace。

        Args:
            None

        Returns:
            dict[str, Any]: {"run_id": ..., "steps": [...]}。

        Raises:
            None
        """
        return {"run_id": self._run_id, "steps": [s.to_dict() for s in self._steps]}

