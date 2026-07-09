"""
数据库引擎与 Session 工厂。

这里提供两个全局对象：
- engine：AsyncEngine，用于连接池和底层连接管理
- SessionLocal：async_sessionmaker，用于在业务代码中创建 session（async with SessionLocal()）
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import get_settings


settings = get_settings()

engine: AsyncEngine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
