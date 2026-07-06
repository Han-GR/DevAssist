from fastapi import FastAPI
from pydantic import BaseModel

import structlog
from uuid import uuid4

from app.core.config import get_settings, setup_logging
from app.core.errors import ConfigurationError, register_error_handlers
from app.core.llm import LLMClient


settings = get_settings()
setup_logging(settings=settings)

logger = structlog.get_logger()
llm_client: LLMClient | None = None

app = FastAPI(title=settings.service_name)
register_error_handlers(app)


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


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    global llm_client

    if llm_client is None:
        try:
            llm_client = LLMClient.from_settings(settings)
        except ValueError as exc:
            raise ConfigurationError(message=str(exc)) from exc

    response = await llm_client.chat(
        messages=[{"role": "user", "content": payload.message}],
        temperature=0.0,
        stream=False,
    )

    content = response.choices[0].message.content if response.choices else ""
    return ChatResponse(reply=content or "")
