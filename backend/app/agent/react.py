from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from app.agent.tools import ToolRegistry
from app.agent.trace import TraceRecorder
from app.core.errors import AppError
from app.core.llm import LLMClient


DEFAULT_MAX_ITERATIONS = 10
MAX_OBSERVATION_CHARS = 8000
TOOL_MAX_RETRIES = 3  # 工具调用失败后最多重试次数（不含首次）


@dataclass(frozen=True)
class ReActStep:
    """
    ReAct 单步记录。

    Args:
        thought (str): 模型输出的思考文本（可能为空）。
        action_raw (str): 模型输出的 Action 行原文（用于排查解析问题）。
        tool_name (str | None): 若为工具调用，则为工具名；否则为 None。
        tool_args (dict[str, Any] | None): 若为工具调用，则为解析出的入参；否则为 None。
        observation (Any | None): 工具输出（或 None）。

    Returns:
        ReActStep: 单步记录对象。

    Raises:
        None
    """

    thought: str
    action_raw: str
    tool_name: str | None
    tool_args: dict[str, Any] | None
    observation: Any | None


class ReActAgent:
    """
    ReAct Agent 最小实现（Thought → Action → Observation 循环）。

    Args:
        llm (LLMClient): LLM 调用封装。
        tools (ToolRegistry): 工具注册表。
        max_iterations (int): 最大迭代次数，默认 10。

    Returns:
        ReActAgent: Agent 实例。

    Raises:
        ValueError: max_iterations 非法时抛出。

    Notes/Examples:
        本实现优先把“闭环跑通”做稳定：
        - 强制解析 Action/args
        - 工具结果以 Observation 形式注入回上下文
        - 达到迭代上限后失败，避免死循环
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._logger = structlog.get_logger()

    async def run(
        self,
        *,
        user_input: str,
        trace: TraceRecorder | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
    ) -> tuple[str, list[ReActStep]]:
        """
        执行一次 ReAct 推理并返回最终答案。

        Args:
            user_input (str): 用户输入。
            trace (TraceRecorder | None): 可选的 trace 记录器；不传则内部新建一个。
            history_messages (Sequence[dict[str, str]] | None): 可选的历史消息（OpenAI-style messages）。

        Returns:
            tuple[str, list[ReActStep]]: (final_answer, steps)

        Raises:
            AppError: 解析失败、达到迭代上限或下游工具/模型异常时抛出。

        Notes/Examples:
            steps 用于后续 trace/可观测性落地；当前阶段先返回给上层调用方。
        """
        if not user_input.strip():
            raise AppError(
                code="agent_input_invalid",
                message="user_input is required.",
                status_code=400,
            )

        system_prompt = _build_system_prompt(tools=self._tools)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(list(history_messages))
        messages.append({"role": "user", "content": user_input})

        steps: list[ReActStep] = []
        recorder = trace or TraceRecorder()

        for i in range(self._max_iterations):
            self._logger.info("react_iteration_start", iteration=i)
            started_at_ms = recorder.start_step(step_index=i)
            resp = await self._llm.chat(messages=list(messages), temperature=0.0, stream=False)
            content = str(resp.choices[0].message.content or "")
            messages.append({"role": "assistant", "content": content})

            try:
                parsed = _parse_react_output(content)
            except AppError as exc:
                recorder.finish_step(
                    step_index=i,
                    started_at_ms=started_at_ms,
                    thought="",
                    action_raw="",
                    tool_name=None,
                    tool_args=None,
                    observation=None,
                    error=str(exc),
                )
                raise

            if parsed["type"] == "final":
                final_answer = parsed["final"]
                steps.append(
                    ReActStep(
                        thought=parsed.get("thought", ""),
                        action_raw=parsed.get("action_raw", ""),
                        tool_name=None,
                        tool_args=None,
                        observation=None,
                    )
                )
                recorder.finish_step(
                    step_index=i,
                    started_at_ms=started_at_ms,
                    thought=parsed.get("thought", ""),
                    action_raw=parsed.get("action_raw", ""),
                    tool_name=None,
                    tool_args=None,
                    observation=None,
                    error=None,
                )
                return final_answer, steps

            tool_name = parsed["tool_name"]
            tool_args = parsed["tool_args"]
            observation, tool_error = await _call_tool_with_retry(
                tools=self._tools,
                tool_name=tool_name,
                tool_args=tool_args,
                max_retries=TOOL_MAX_RETRIES,
                logger=self._logger,
            )
            recorder.finish_step(
                step_index=i,
                started_at_ms=started_at_ms,
                thought=parsed.get("thought", ""),
                action_raw=parsed.get("action_raw", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation,
                error=tool_error,
            )
            steps.append(
                ReActStep(
                    thought=parsed.get("thought", ""),
                    action_raw=parsed.get("action_raw", ""),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                )
            )

            if tool_error is not None:
                # 优雅降级：把错误作为 Observation 注入，让模型决定下一步
                observation_text = _format_tool_error_observation(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    error=tool_error,
                )
            else:
                observation_text = _format_tool_observation(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                )
            messages.append({"role": "user", "content": observation_text})

        raise AppError(
            code="agent_max_iterations",
            message="Agent reached max iterations without producing a final answer.",
            status_code=408,
            details={"max_iterations": self._max_iterations},
        )


def _build_system_prompt(*, tools: ToolRegistry) -> str:
    """
    构建 ReAct Agent 的 system prompt（含工具清单与输出格式约定）。

    Args:
        tools (ToolRegistry): 工具注册表。

    Returns:
        str: system prompt 文本。

    Raises:
        None

    Notes/Examples:
        这里用“尽量稳定、可解析”的格式，而不是追求最短提示词。
    """
    tool_lines: list[str] = []
    for tool in tools.list():
        schema = json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True)
        tool_lines.append(f"- {tool.name}: {tool.description}\n  parameters: {schema}")
    tool_block = "\n".join(tool_lines) if tool_lines else "- (no tools)"

    return "\n".join(
        [
            "你是 DevAssist 的 ReAct Agent，擅长把复杂问题拆成可执行步骤，并调用工具获取证据。",
            "",
            "可用工具：",
            tool_block,
            "",
            "输出格式要求：",
            "- 你必须输出两段：Thought 与 Action",
            "- Action 只有两种：",
            "  1) tool call:",
            "     Action: tool:<tool_name>",
            "     args: <json object>",
            "  2) final answer:",
            "     Action: final: <final answer text>",
            "",
            "注意：",
            "- args 必须是严格 JSON（双引号），不要输出多余字段",
            "- 每次 tool call 后，我会把 Observation（含 tool_name/tool_args/result）发给你，你再继续下一步",
        ]
    )


_RE_ACTION_LINE = re.compile(r"^\s*Action\s*:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_ARGS_LINE = re.compile(r"^\s*args\s*:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_THOUGHT_LINE = re.compile(r"^\s*Thought\s*:\s*(.*)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_react_output(text: str) -> dict[str, Any]:
    """
    解析模型输出（Thought/Action/args）。

    Args:
        text (str): 模型输出文本。

    Returns:
        dict[str, Any]:
            - type: "tool" | "final"
            - thought: str
            - action_raw: str
            - tool_name/tool_args 或 final

    Raises:
        AppError: 解析失败时抛出。
    """
    action_match = _RE_ACTION_LINE.search(text)
    if action_match is None:
        raise AppError(
            code="agent_parse_error",
            message="Missing Action line in model output.",
            status_code=500,
            details={"output": text[:2000]},
        )
    action_raw = action_match.group(1).strip()

    thought_match = _RE_THOUGHT_LINE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    lowered = action_raw.lower()
    if lowered.startswith("final:"):
        final_inline = action_raw[len("final:") :].lstrip()
        if final_inline:
            return {"type": "final", "final": final_inline, "thought": thought, "action_raw": action_raw}

        after_action = text[action_match.end() :].strip()
        return {"type": "final", "final": after_action, "thought": thought, "action_raw": action_raw}

    if "tool:" not in lowered:
        raise AppError(
            code="agent_parse_error",
            message="Action must be a tool call or final answer.",
            status_code=500,
            details={"action": action_raw},
        )

    tool_name = action_raw.split("tool:", 1)[1].strip()
    if not tool_name:
        raise AppError(
            code="agent_parse_error",
            message="Tool name is missing in Action line.",
            status_code=500,
            details={"action": action_raw},
        )

    args_match = _RE_ARGS_LINE.search(text)
    if args_match is None:
        raise AppError(
            code="agent_parse_error",
            message="Missing args line for tool call.",
            status_code=500,
            details={"action": action_raw},
        )

    raw_args = args_match.group(1).strip()
    if raw_args.startswith("```"):
        raw_args = text[args_match.start(1) :].strip()
        raw_args = _strip_code_fence(raw_args)

    try:
        tool_args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise AppError(
            code="agent_parse_error",
            message="Tool args must be valid JSON.",
            status_code=500,
            details={"error": str(exc), "raw_args": raw_args[:2000]},
        ) from exc

    if not isinstance(tool_args, dict):
        raise AppError(
            code="agent_parse_error",
            message="Tool args must be a JSON object.",
            status_code=500,
            details={"raw_args": raw_args[:2000]},
        )

    return {
        "type": "tool",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "thought": thought,
        "action_raw": action_raw,
    }


def _strip_code_fence(text: str) -> str:
    """
    去除可能的 Markdown 代码块包裹。

    Args:
        text (str): 可能以 ``` 开头的文本。

    Returns:
        str: 去除 fence 后的纯内容。

    Raises:
        None
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) <= 2:
        return ""
    return "\n".join(lines[1:-1]).strip()


def _safe_json_dumps(value: Any) -> str:
    """
    将任意对象尽量序列化为 JSON 字符串。

    Args:
        value (Any): 任意对象。

    Returns:
        str: JSON 字符串；若无法 JSON 序列化，则退化为 str(value)。

    Raises:
        None
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


async def _call_tool_with_retry(
    *,
    tools: ToolRegistry,
    tool_name: str,
    tool_args: dict[str, Any],
    max_retries: int,
    logger: Any,
) -> tuple[Any, str | None]:
    """
    调用工具，失败后最多重试 max_retries 次。

    Args:
        tools (ToolRegistry): 工具注册表。
        tool_name (str): 工具名称。
        tool_args (dict[str, Any]): 工具入参。
        max_retries (int): 最大重试次数（不含首次调用）。
        logger: structlog logger 实例。

    Returns:
        tuple[Any, str | None]: (observation, error_message)
            - 成功时 observation 为工具返回值，error_message 为 None
            - 全部失败时 observation 为 None，error_message 为最后一次异常信息

    Raises:
        None: 所有异常均被捕获，通过返回值传递。

    Notes/Examples:
        重试间隔为 0（不 sleep），保持 Agent 响应速度；
        每次失败都会打结构化日志 tool_call_failed。
    """
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await tools.call(name=tool_name, payload=tool_args)
            if attempt > 0:
                logger.info(
                    "tool_call_retry_success",
                    tool_name=tool_name,
                    attempt=attempt,
                )
            return result, None
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "tool_call_failed",
                tool_name=tool_name,
                attempt=attempt,
                max_retries=max_retries,
                error=last_error,
            )
    return None, last_error


def _format_tool_error_observation(
    *, tool_name: str, tool_args: dict[str, Any], error: str
) -> str:
    """
    将工具调用失败信息格式化为 Observation 文本，供模型决定下一步。

    Args:
        tool_name (str): 工具名称。
        tool_args (dict[str, Any]): 工具入参。
        error (str): 错误信息。

    Returns:
        str: Observation 文本（包含 tool_name/tool_args/error/suggestion）。

    Raises:
        None
    """
    payload_obj: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "error": error,
        "suggestion": "The tool call failed after retries. You may try a different approach or provide a final answer based on what you know.",
    }
    return f"Observation:\n{_safe_json_dumps(payload_obj)}"


def _format_tool_observation(*, tool_name: str, tool_args: dict[str, Any], observation: Any) -> str:
    """
    将工具调用结果格式化为 Observation 文本，供下一轮模型推理使用。

    Args:
        tool_name (str): 工具名称。
        tool_args (dict[str, Any]): 工具入参。
        observation (Any): 工具输出。

    Returns:
        str: Observation 文本（包含 tool_name/tool_args/result），并对超长内容做截断保护。

    Raises:
        None

    Notes/Examples:
        为了保证“可解析 + 不炸上下文”，Observation 的内容优先使用 JSON；
        超过长度上限时会返回 result_preview + result_truncated 标记。
    """
    payload_obj: dict[str, Any] = {"tool_name": tool_name, "tool_args": tool_args, "result": observation}
    payload = _safe_json_dumps(payload_obj)
    if len(payload) <= MAX_OBSERVATION_CHARS:
        return f"Observation:\n{payload}"

    preview = _safe_json_dumps(observation)
    if len(preview) > MAX_OBSERVATION_CHARS:
        preview = preview[:MAX_OBSERVATION_CHARS]
    truncated_obj = {
        "tool_name": tool_name,
        "tool_args": tool_args,
        "result_preview": preview,
        "result_truncated": True,
    }
    return f"Observation:\n{_safe_json_dumps(truncated_obj)}"
