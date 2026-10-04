"""Функции для работы с базой данных (CRUD для WardrobeItem)."""

import logging
from typing import Sequence

from sqlalchemy import delete, select

from app.db.database import SessionLocal
from app.db.models import WardrobeItem

logger = logging.getLogger(__name__)


async def add_item(
    *,
    user_id: int,
    photo_file_id: str,
    category: str,
    type: str,
    color: str = "",
    material: str = "",
    warmth_level: int = 3,
    waterproof: bool = False,
    formal_level: str = "casual",
    season: str = "универсальная",
) -> WardrobeItem:
    """Добавить вещь в гардероб. Возвращает созданный объект с id."""
    async with SessionLocal() as session:
        item = WardrobeItem(
            user_id=user_id,
            photo_file_id=photo_file_id,
            category=category,
            type=type,
            color=color,
            material=material,
            warmth_level=warmth_level,
            waterproof=waterproof,
            formal_level=formal_level,
            season=season,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        logger.info("Добавлена вещь id=%s user_id=%s", item.id, user_id)
        return item


async def get_items(user_id: int) -> Sequence[WardrobeItem]:
    """Получить все вещи пользователя."""
    async with SessionLocal() as session:
        result = await session.execute(
            select(WardrobeItem)
            .where(WardrobeItem.user_id == user_id)
            .order_by(WardrobeItem.created_at.desc())
        )
        return result.scalars().all()


async def get_item(item_id: int) -> WardrobeItem | None:
    """Получить вещь по id."""
    async with SessionLocal() as session:
        return await session.get(WardrobeItem, item_id)


async def delete_item(item_id: int, user_id: int) -> bool:
    """Удалить вещь. Возвращает True, если что-то было удалено.

    Проверка user_id не даёт удалить чужую вещь.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            delete(WardrobeItem).where(
                WardrobeItem.id == item_id,
                WardrobeItem.user_id == user_id,
            )
        )
        await session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("Удалена вещь id=%s user_id=%s", item_id, user_id)
        return deleted


async def count_items(user_id: int) -> int:
    """Сколько вещей у пользователя."""
    async with SessionLocal() as session:
        result = await session.execute(
            select(WardrobeItem).where(WardrobeItem.user_id == user_id)
        )
        return len(result.scalars().all())


async def update_item_field(
    item_id: int, user_id: int, field: str, value
) -> bool:
    """Обновить одно поле вещи. Возвращает True, если обновилось.

    Разрешены только поля из белого списка — нельзя случайно
    перезаписать id, user_id или created_at.
    """
    allowed = {
        "category", "type", "color", "warmth_level",
        "waterproof", "formal_level", "season",
    }
    if field not in allowed:
        logger.warning("Попытка обновить запрещённое поле: %s", field)
        return False

    async with SessionLocal() as session:
        result = await session.execute(
            select(WardrobeItem).where(
                WardrobeItem.id == item_id,
                WardrobeItem.user_id == user_id,
            )
        )
        item = result.scalar_one_or_none()
        if item is None:
            return False

        setattr(item, field, value)
        await session.commit()
        logger.info(
            "Обновлена вещь id=%s user_id=%s: %s=%r",
            item_id, user_id, field, value,
        )
        return True