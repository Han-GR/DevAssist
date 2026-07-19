"""
FastAPI 应用入口。

这个文件刻意保持“薄”：
- 做全局初始化（Settings、日志、异常处理器）
- 注册路由
- 加 request_id 中间件（把一次请求串起来）
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import time
import structlog
from uuid import uuid4

from app.core.config import get_settings, setup_logging
from app.core.errors import register_error_handlers
from app.core.rate_limit import RateLimiter
from app.api.agent import router as agent_router
from app.api.admin_evals import router as admin_evals_router
from app.api.admin_traces import router as admin_traces_router
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.search import router as search_router


settings = get_settings()
setup_logging(settings=settings)

logger = structlog.get_logger()
rate_limiter = RateLimiter.from_settings(settings=settings)

app = FastAPI(title=settings.service_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_error_handlers(app)
app.include_router(agent_router)
app.include_router(admin_evals_router)
app.include_router(admin_traces_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(search_router)


@app.middleware("http")
async def bind_request_id(request, call_next):
    """
    为每个请求绑定 request_id，并写回到响应头。

    Args:
        request: FastAPI/Starlette 的 Request 对象。
        call_next: 下一个中间件/路由处理函数。

    Returns:
        Response: 原始响应对象（会额外带上 x-request-id 响应头）。

    Raises:
        Exception: 下游处理抛出的异常会交由全局异常处理器统一处理。

    Notes:
        request_id 会进入 structlog 的 contextvars，后续任何日志都能自动带上它；
        排查问题时不用在一堆日志里猜“哪几行属于同一次请求”。
    """
    # 方便串联一次请求在各处的日志；上游如果已经带了 x-request-id，就沿用它
    request_id = request.headers.get("x-request-id") or str(uuid4())
    user_id = request.headers.get("x-user-id")
    started_at = time.perf_counter()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, user_id=user_id)

    if (
        rate_limiter is not None
        and request.method != "OPTIONS"
        and str(request.url.path) not in {"/health", "/docs", "/openapi.json"}
    ):
        from starlette.responses import JSONResponse

        identity = user_id or getattr(request.client, "host", None) or "unknown"
        structlog.contextvars.bind_contextvars(rate_limit_identity=identity)
        decision = await rate_limiter.check(identity=identity)
        if not decision.allowed:
            message = "Too many requests. Please try again later."
            logger.info(
                "app_error",
                code="rate_limited",
                status_code=429,
                message=message,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": message,
                        "details": {
                            "limit": decision.limit,
                            "window_seconds": decision.window_seconds,
                        },
                    },
                    "request_id": request_id,
                },
            )
            response.headers["x-request-id"] = request_id
            response.headers["retry-after"] = str(decision.window_seconds)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info(
                "http_request",
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=user_id,
            )
            structlog.contextvars.clear_contextvars()
            return response

    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        duration_ms = int((time.perf_counter() - started_at) * 1000)

        logger.info(
            "http_request",
            method=request.method,
            path=str(request.url.path),
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
        )
        structlog.contextvars.clear_contextvars()


@app.get("/health")
async def health() -> dict[str, str]:
    """
    健康检查接口。

    Returns:
        dict[str, str]: 固定返回 {"status": "ok"}。

    Notes:
        这个接口尽量不依赖外部资源（DB/LLM），只要进程活着、路由能跑通就返回 ok，
        便于容器和监控系统做存活探测。
    """
    return {"status": "ok"}
