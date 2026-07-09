"""
FastAPI 应用入口。

这个文件刻意保持“薄”：
- 做全局初始化（Settings、日志、异常处理器）
- 注册路由
- 加 request_id 中间件（把一次请求串起来）
"""

from fastapi import FastAPI

import structlog
from uuid import uuid4

from app.core.config import get_settings, setup_logging
from app.core.errors import register_error_handlers
from app.api.chat import router as chat_router


settings = get_settings()
setup_logging(settings=settings)

logger = structlog.get_logger()

app = FastAPI(title=settings.service_name)
register_error_handlers(app)
app.include_router(chat_router)


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

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)
    response.headers["x-request-id"] = request_id

    # 这里先做最基础的访问日志，后面接入 DB/LLM 调用时也可以继续复用 request_id
    logger.info(
        "http_request",
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
    )

    structlog.contextvars.clear_contextvars()
    return response


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
