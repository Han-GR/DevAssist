from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

import structlog

from app.core.errors import AppError


JSONSchema = dict[str, Any]
ToolHandler = Callable[..., Any] | Callable[..., Awaitable[Any]]


def _is_instance_of_type(value: Any, schema_type: str) -> bool:
    """
    判断 Python 值是否符合 JSON Schema 的 type。

    Args:
        value (Any): 待判断的值。
        schema_type (str): JSON Schema 的 type 字段。

    Returns:
        bool: 是否符合。

    Raises:
        None

    Notes/Examples:
        这里实现的是“够用的最小子集”，避免引入额外依赖：
        - object/array/string/integer/number/boolean/null
        - bool 在 Python 中是 int 子类，需要优先判断 boolean。
    """
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return False


def _validate_schema_definition(*, schema: JSONSchema, path: str = "$") -> None:
    """
    校验 JSON Schema 的结构是否是“本项目支持的子集”。

    Args:
        schema (JSONSchema): JSON Schema 定义。
        path (str): 递归校验时的路径标识。

    Returns:
        None: 校验通过不返回。

    Raises:
        ValueError: schema 结构不合法或包含不支持的字段时抛出。

    Notes/Examples:
        目标是尽早在“注册工具”阶段暴露 schema 错误，而不是运行时才发现。
    """
    if not isinstance(schema, dict):
        raise ValueError(f"schema must be a dict, got {type(schema).__name__} at {path}")

    schema_type = schema.get("type")
    if schema_type not in {
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }:
        raise ValueError(f"schema.type is required and must be a known JSON type at {path}")

    if schema_type == "object":
        props = schema.get("properties", {})
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise ValueError(f"schema.properties must be a dict at {path}")
        for k, v in props.items():
            if not isinstance(k, str):
                raise ValueError(f"schema.properties key must be a string at {path}")
            _validate_schema_definition(schema=v, path=f"{path}.properties.{k}")

        required = schema.get("required", [])
        if required is None:
            required = []
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            raise ValueError(f"schema.required must be a list[str] at {path}")
        if any(x not in props for x in required):
            raise ValueError(f"schema.required contains unknown property at {path}")

        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool):
            raise ValueError(f"schema.additionalProperties must be a bool at {path}")
        return

    if schema_type == "array":
        items = schema.get("items")
        if items is None:
            raise ValueError(f"schema.items is required for array at {path}")
        _validate_schema_definition(schema=items, path=f"{path}.items")
        return


def _validate_instance_against_schema(
    *, value: Any, schema: JSONSchema, path: str = "$"
) -> None:
    """
    校验值是否符合 schema（本项目支持的 JSON Schema 子集）。

    Args:
        value (Any): 待校验的值。
        schema (JSONSchema): JSON Schema 定义。
        path (str): 当前校验路径。

    Returns:
        None: 校验通过不返回。

    Raises:
        AppError: 校验失败时抛出（code=tool_input_invalid）。

    Notes/Examples:
        这是工具调用前的“入参校验层”，用于在 Agent 侧尽早拦截明显错误输入。
    """
    schema_type = str(schema.get("type"))
    if not _is_instance_of_type(value, schema_type):
        raise AppError(
            code="tool_input_invalid",
            message="Tool input does not match schema.",
            status_code=400,
            details={"path": path, "expected_type": schema_type, "actual_type": type(value).__name__},
        )

    if schema_type == "object":
        props: dict[str, JSONSchema] = dict(schema.get("properties") or {})
        required: list[str] = list(schema.get("required") or [])
        additional: bool = bool(schema.get("additionalProperties", True))

        for k in required:
            if k not in value:
                raise AppError(
                    code="tool_input_invalid",
                    message="Tool input does not match schema.",
                    status_code=400,
                    details={"path": f"{path}.{k}", "error": "missing_required"},
                )

        for k, v in value.items():
            if k in props:
                _validate_instance_against_schema(value=v, schema=props[k], path=f"{path}.{k}")
                continue
            if not additional:
                raise AppError(
                    code="tool_input_invalid",
                    message="Tool input does not match schema.",
                    status_code=400,
                    details={"path": f"{path}.{k}", "error": "additional_property_not_allowed"},
                )
        return

    if schema_type == "array":
        items_schema = schema.get("items") or {}
        for i, item in enumerate(value):
            _validate_instance_against_schema(value=item, schema=items_schema, path=f"{path}[{i}]")
        return


@dataclass(frozen=True)
class Tool:
    """
    Agent 可调用工具的最小定义。

    Args:
        name (str): 工具唯一名称（用于模型输出与路由匹配）。
        description (str): 工具用途说明（给模型看）。
        parameters (JSONSchema): 工具入参 JSON Schema（建议 type=object）。
        handler (ToolHandler): 实际执行逻辑，支持同步或异步函数。
        return_schema (JSONSchema | None): 可选的返回值 schema（当前主要用于文档/约束表达）。

    Returns:
        Tool: 工具实例。

    Raises:
        ValueError: name/description/schema 不合法时抛出。

    Notes/Examples:
        这个结构设计为“足够小，但能支撑后续 ReAct + function calling”：
        - name/description/parameters 用于向 LLM 暴露工具能力
        - handler 用于执行工具
        - validate_input 用于在执行前做入参校验，保证可控性
    """

    name: str
    description: str
    parameters: JSONSchema
    handler: ToolHandler
    return_schema: JSONSchema | None = None

    def __post_init__(self) -> None:
        """
        初始化后校验 Tool 定义的基本合法性。

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: 当字段缺失或 schema 结构不合法时抛出。

        Notes/Examples:
            这里抛 ValueError 是“开发期错误”，用于尽早暴露工具定义问题。
        """
        if not self.name.strip():
            raise ValueError("tool.name is required")
        if not self.description.strip():
            raise ValueError("tool.description is required")
        _validate_schema_definition(schema=self.parameters, path=f"tool:{self.name}.parameters")
        if self.return_schema is not None:
            _validate_schema_definition(schema=self.return_schema, path=f"tool:{self.name}.return_schema")

    def to_openai_tool(self) -> dict[str, Any]:
        """
        转换为 OpenAI-style 的 tools/function calling 结构。

        Args:
            None

        Returns:
            dict[str, Any]: OpenAI tools 数组中的单个元素结构。

        Raises:
            None

        Notes/Examples:
            形状为：
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_input(self, payload: Mapping[str, Any]) -> None:
        """
        校验工具输入 payload 是否符合 parameters schema。

        Args:
            payload (Mapping[str, Any]): 工具入参（通常是 dict）。

        Returns:
            None: 校验通过不返回。

        Raises:
            AppError: payload 不符合 schema 时抛出。
        """
        if not isinstance(payload, dict):
            raise AppError(
                code="tool_input_invalid",
                message="Tool input must be an object.",
                status_code=400,
                details={"expected_type": "object", "actual_type": type(payload).__name__},
            )
        _validate_instance_against_schema(value=payload, schema=self.parameters, path="$")

    async def call(self, payload: Mapping[str, Any]) -> Any:
        """
        校验并执行工具。

        Args:
            payload (Mapping[str, Any]): 工具入参（dict）。

        Returns:
            Any: handler 的返回值。

        Raises:
            AppError: payload 不符合 schema 时抛出。
            Exception: handler 执行异常会原样抛出，由上层统一处理。

        Notes/Examples:
            为了方便后续工具实现，handler 默认按 kwargs 方式接收参数。
        """
        self.validate_input(payload)
        if not isinstance(payload, dict):
            raise AppError(
                code="tool_input_invalid",
                message="Tool input must be an object.",
                status_code=400,
                details={"expected_type": "object", "actual_type": type(payload).__name__},
            )

        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**payload)
        result = self.handler(**payload)
        if asyncio.iscoroutine(result):
            return await result
        return result


class ToolRegistry:
    """
    Tool 注册表。

    目标：
    - 统一管理工具集合（注册/查询/枚举）
    - 对 name 冲突、schema 不合法等问题提前失败
    """

    def __init__(self) -> None:
        """
        初始化注册表。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self._tools: dict[str, Tool] = {}
        self._logger = structlog.get_logger()

    def register(self, tool: Tool) -> None:
        """
        注册一个工具。

        Args:
            tool (Tool): 待注册的工具。

        Returns:
            None

        Raises:
            AppError: 工具名冲突时抛出。

        Notes/Examples:
            schema 合法性已由 Tool.__post_init__ 保证；这里主要处理 name 冲突。
        """
        if tool.name in self._tools:
            raise AppError(
                code="tool_already_registered",
                message="Tool is already registered.",
                status_code=409,
                details={"name": tool.name},
            )
        self._tools[tool.name] = tool
        self._logger.info("tool_registered", name=tool.name)

    def get(self, name: str) -> Tool:
        """
        按名称获取工具。

        Args:
            name (str): 工具名称。

        Returns:
            Tool: 工具实例。

        Raises:
            AppError: 工具不存在时抛出。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise AppError(
                code="tool_not_found",
                message="Tool not found.",
                status_code=404,
                details={"name": name},
            )
        return tool

    def list(self) -> list[Tool]:
        """
        列出所有已注册工具。

        Args:
            None

        Returns:
            list[Tool]: 工具列表（按注册顺序）。

        Raises:
            None
        """
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        导出 OpenAI-style tools 列表。

        Args:
            None

        Returns:
            list[dict[str, Any]]: tools 列表，可直接传给 OpenAI/DeepSeek 的 tools 参数。

        Raises:
            None
        """
        return [t.to_openai_tool() for t in self.list()]

    async def call(self, *, name: str, payload: Mapping[str, Any]) -> Any:
        """
        通过注册表调用指定工具。

        Args:
            name (str): 工具名称。
            payload (Mapping[str, Any]): 工具入参。

        Returns:
            Any: 工具返回值。

        Raises:
            AppError: 工具不存在或输入不合法时抛出。
            Exception: 工具执行异常原样抛出，由上层统一处理。
        """
        tool = self.get(name)
        start = time.perf_counter()
        try:
            result = await tool.call(payload)
            self._logger.info(
                "tool_call",
                name=name,
                success=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
            return result
        except Exception as exc:
            self._logger.exception(
                "tool_call",
                name=name,
                success=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=str(exc),
            )
            raise

