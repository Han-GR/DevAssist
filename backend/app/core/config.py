from __future__ import annotations

from functools import lru_cache
from typing import Literal

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog


class Settings(BaseSettings):
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    service_name: str = "devassist-backend"

    model_config = SettingsConfigDict(
        # 开发阶段优先图省事：本地可以放一个 .env；线上则直接走环境变量
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(*, settings: Settings) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    if not root_logger.handlers:
        # 避免重复添加 handler（比如热重载、或测试里多次初始化）
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            # 统一输出 JSON，后面接 Loki/ELK 会更省心
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
