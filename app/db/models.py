"""SQLAlchemy-модели базы данных."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс для всех моделей. Хранит метаданные таблиц."""
    pass


class WardrobeItem(Base):
    """Вещь в гардеробе пользователя."""

    __tablename__ = "wardrobe_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Telegram ID пользователя. Храним как Integer — Telegram использует
    # положительные числа до ~10^12, помещается в 64-битный int.
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    # file_id фотографии в Telegram. Позволяет повторно отправлять фото
    # без скачивания и повторной загрузки.
    photo_file_id: Mapped[str] = mapped_column(String(256), nullable=False)

    # Категория: верх / низ / обувь / аксессуар
    category: Mapped[str] = mapped_column(String(32), nullable=False)

    # Тип: футболка, джинсы, куртка и т.д.
    type: Mapped[str] = mapped_column(String(64), nullable=False)

    # Основной цвет (например, "белый", "чёрный")
    color: Mapped[str] = mapped_column(String(32), default="")

    # Материал (опционально)
    material: Mapped[str] = mapped_column(String(64), default="")

    # Уровень тепла 1–5 (1 — очень лёгкая, 5 — очень тёплая)
    warmth_level: Mapped[int] = mapped_column(Integer, default=3)

    # Водонепроницаемость
    waterproof: Mapped[bool] = mapped_column(Boolean, default=False)

    # Формальность: casual / business / sport
    formal_level: Mapped[str] = mapped_column(String(32), default="casual")

    # Сезон: лето / демисезон / зима / универсальная
    season: Mapped[str] = mapped_column(String(32), default="универсальная")

    # Дата добавления
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<WardrobeItem id={self.id} user_id={self.user_id} "
            f"category={self.category!r} type={self.type!r}>"
        )