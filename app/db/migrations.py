"""Простые миграции для SQLite: добавляем недостающие колонки."""

import logging
from sqlalchemy import text
from app.db.database import engine

logger = logging.getLogger(__name__)

NEW_COLUMNS = [
    ("accessory_occasions", "TEXT", "'[]'"),
]

async def run_migrations() -> None:
    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(wardrobe_items)"))
        existing = {row[1] for row in result.fetchall()}

        for name, sql_type, default in NEW_COLUMNS:
            if name in existing:
                continue
            sql = f"ALTER TABLE wardrobe_items ADD COLUMN {name} {sql_type} DEFAULT {default}"
            logger.info("Миграция: %s", sql)
            await conn.execute(text(sql))
    logger.info("Миграции завершены")
