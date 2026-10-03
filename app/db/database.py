"""Подключение к базе данных SQLite через SQLAlchemy (async)."""

import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)

# Движок. echo=False — не засоряем логи SQL-запросами.
# Для отладки можно поставить echo=True.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

# Фабрика сессий. expire_on_commit=False — чтобы объекты оставались
# доступны после коммита (по умолчанию SQLAlchemy их "забывает").
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Создать таблицы, если их ещё нет.

    Для MVP используем этот подход. Когда схема начнёт меняться,
    перейдём на Alembic (миграции).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("База данных инициализирована: %s", settings.DATABASE_URL)