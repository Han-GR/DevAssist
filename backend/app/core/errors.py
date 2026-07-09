"""
统一错误处理。

这个模块的目标是“让接口的错误形状稳定”：不管是业务异常、参数校验失败，还是未知异常，
最终都返回一套一致的 JSON 结构，并携带 request_id 方便排查。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
import structlog
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse


class AppError(Exception):
    """
    业务层可预期异常的基类。

    code：给前端/调用方做分支处理
    message：给人看的信息
    details：留给调试（可为空）
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 400,
        details: Any | None = None,
    ) -> None:
        """
        初始化业务异常。

        Args:
            code (str): 业务错误码，供前端/调用方做分支处理。
            message (str): 面向用户/开发者的错误描述。
            status_code (int): HTTP 状态码，默认 400。
            details (Any | None): 可选的额外调试信息（比如字段级错误、上下文信息等）。

        Returns:
            None: 只初始化异常对象。

        Raises:
            None

        Notes:
            这里把 status_code 也放进来，是因为业务错误并不总是 400：
            - 404（资源不存在）
            - 409（冲突）
            - 429（限流）
            等都可能是“业务可预期”。
        """
        # 业务错误统一走这里：code 给前端/调用方做分支处理，message 给人看，details 留给调试
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ConfigurationError(AppError):
    """
    配置相关错误（通常是服务端问题）。

    比如：LLM key 没配、模型名为空、base_url 不合法等。
    """

    def __init__(self, *, message: str, details: Any | None = None) -> None:
        """
        初始化配置错误。

        Args:
            message (str): 错误描述。
            details (Any | None): 额外调试信息。

        Returns:
            None

        Raises:
            None

        Notes:
            这类错误对用户来说通常是“服务端不可用”，所以统一按 500 返回。
        """
        super().__init__(
            code="configuration_error",
            message=message,
            status_code=500,
            details=details,
        )


def register_error_handlers(app: FastAPI) -> None:
    """
    为 FastAPI 应用注册全局异常处理器。

    Args:
        app (FastAPI): FastAPI 应用实例。

    Returns:
        None: 仅注册 handler，不返回业务数据。

    Raises:
        None

    Notes:
        这里用“集中注册”的方式，避免每个路由里写一套 try/except，
        也便于保证所有接口返回的错误 JSON 结构一致。
    """
    # 把错误处理集中在这里注册，避免散落在各个路由里
    logger = structlog.get_logger()

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """
        处理业务异常（AppError）。

        Args:
            request (Request): 当前请求对象（主要用于保持签名一致）。
            exc (AppError): 业务异常实例。

        Returns:
            JSONResponse: 统一错误格式的 JSON 响应。

        Raises:
            None

        Notes:
            - 4xx：可预期，记录 info
            - 5xx：服务端问题，记录 exception（带堆栈）
        """
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        # 5xx 认为是服务端问题，打 exception；4xx 属于业务可预期问题，info 就够了
        if exc.status_code >= 500:
            logger.exception(
                "app_error",
                code=exc.code,
                status_code=exc.status_code,
                message=exc.message,
            )
        else:
            logger.info(
                "app_error",
                code=exc.code,
                status_code=exc.status_code,
                message=exc.message,
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
                "request_id": request_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        处理请求参数校验错误（FastAPI/Pydantic 自动抛出）。

        Args:
            request (Request): 当前请求对象（主要用于保持签名一致）。
            exc (RequestValidationError): 参数校验错误详情。

        Returns:
            JSONResponse: 统一错误格式的 422 JSON 响应。

        Raises:
            None

        Notes:
            这类错误通常是：
            - 缺字段
            - 类型不匹配（比如 UUID 解析失败）
        """
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        # 这里就是 Pydantic/FastAPI 兜底的参数校验错误（缺字段、类型不对之类）
        logger.info("validation_error", errors=exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request",
                    "details": exc.errors(),
                },
                "request_id": request_id,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        处理框架层的 HTTPException（比如 404/401）。

        Args:
            request (Request): 当前请求对象（主要用于保持签名一致）。
            exc (StarletteHTTPException): Starlette/FastAPI 抛出的 HTTP 异常。

        Returns:
            JSONResponse: 统一错误格式的 JSON 响应。

        Raises:
            None

        Notes:
            这类异常本质上也应该走统一错误格式，避免前端需要兼容两种返回结构。
        """
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        # FastAPI/Starlette 自己抛的 HTTPException（比如 404、401），统一成 JSON 格式
        logger.info("http_exception", status_code=exc.status_code, detail=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_exception",
                    "message": str(exc.detail),
                    "details": None,
                },
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """
        真正的兜底处理器：捕获所有未预期异常。

        Args:
            request (Request): 当前请求对象（主要用于保持签名一致）。
            exc (Exception): 未捕获的异常实例。

        Returns:
            JSONResponse: 统一错误格式的 500 JSON 响应（不暴露内部细节）。

        Raises:
            None

        Notes:
            返回值不暴露内部细节；堆栈信息只留在日志里，避免把敏感信息直接抛给用户。
        """
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        # 真正兜底的异常：不给用户看内部细节，日志里会带堆栈，排查用
        logger.exception("unhandled_exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "details": None,
                },
                "request_id": request_id,
            },
        )
