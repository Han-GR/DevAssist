"""
配置与日志初始化。

项目早期把配置集中在一个 Settings 里，目的是：
- 本地用 .env 快速启动；
- 线上用环境变量覆盖；
- 代码里尽量不要到处 os.getenv，避免配置来源分散。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog


class Settings(BaseSettings):
    """
    后端配置项。

    这里的字段会自动从环境变量读取（pydantic-settings），并提供默认值，
    方便本地开发直接跑起来。
    """

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    service_name: str = "devassist-backend"
    llm_provider: Literal["deepseek", "openai"] = "deepseek"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "devassist"
    database_url: str = "postgresql+asyncpg://devassist:devassist@db:5432/devassist"

    model_config = SettingsConfigDict(
        # 开发阶段优先图省事：本地可以放一个 .env；线上则直接走环境变量
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    获取 Settings 单例。

    Returns:
        Settings: 当前进程内缓存的配置对象。

    Raises:
        Exception: 当环境变量/配置值不合法时，pydantic-settings 可能抛出校验异常。

    Notes:
        加 lru_cache 的主要原因是：在一个进程里 Settings 不需要重复解析，
        也避免在热路径里反复读 env/.env。测试里如果需要重新读取配置，要手动 cache_clear。
    """
    return Settings()


def setup_logging(*, settings: Settings) -> None:
    """
    初始化 structlog（JSON 日志）。

    Args:
        settings (Settings): 配置对象，主要使用其中的 log_level。

    Returns:
        None: 只做初始化配置。

    Raises:
        Exception: 极少数情况下（例如 logging handler 初始化失败）可能抛出异常。

    Notes:
        我们让根 logger 只输出纯 message，再交给 structlog 的 processors 统一渲染成 JSON，
        后续接 Loki/ELK 时会省很多麻烦。
    """
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
