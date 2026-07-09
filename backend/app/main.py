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
    return {"status": "ok"}
