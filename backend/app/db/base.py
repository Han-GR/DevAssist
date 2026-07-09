"""
SQLAlchemy Declarative Base 定义。

把 Base 单独抽出来的目的，是让 models.py 只关心“表怎么长什么样”，
同时也方便 Alembic 在导入模型时拿到统一的 metadata。
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    ORM 模型基类。

    当前我们不在这里塞太多“魔法”，保持干净；
    如果后面需要全局通用字段或行为，再逐步加也不迟。
    """

    pass
