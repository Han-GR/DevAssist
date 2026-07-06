from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
import structlog
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse


class AppError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 400,
        details: Any | None = None,
    ) -> None:
        # 业务错误统一走这里：code 给前端/调用方做分支处理，message 给人看，details 留给调试
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ConfigurationError(AppError):
    def __init__(self, *, message: str, details: Any | None = None) -> None:
        super().__init__(
            code="configuration_error",
            message=message,
            status_code=500,
            details=details,
        )


def register_error_handlers(app: FastAPI) -> None:
    # 把错误处理集中在这里注册，避免散落在各个路由里
    logger = structlog.get_logger()

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
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
