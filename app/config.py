"""Загрузка конфигурации из переменных окружения (.env)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Загружаем переменные из .env в окружение процесса
load_dotenv()


# Hugging Face Spaces монтирует постоянное хранилище в /data
if os.path.exists("/data"):
    DATABASE_URL_DEFAULT = "sqlite+aiosqlite:////data/wardrobe.db"
else:
    DATABASE_URL_DEFAULT = "sqlite+aiosqlite:///./data/wardrobe.db"


@dataclass(frozen=True)
class Settings:
    """Настройки приложения. Значения читаются из .env один раз при импорте."""

    BOT_TOKEN: str
    HF_TOKEN: str
    OWM_API_KEY: str
    DATABASE_URL: str


def _get_env(name: str, default: str = "") -> str:
    """Вернуть значение переменной окружения или default."""
    return os.getenv(name, default).strip()


def load_settings() -> Settings:
    """Собрать настройки из окружения и проверить обязательные поля."""
    bot_token = _get_env("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Проверьте файл .env в корне проекта."
        )

    return Settings(
        BOT_TOKEN=bot_token,
        HF_TOKEN=_get_env("HF_TOKEN"),
        OWM_API_KEY=_get_env("OWM_API_KEY"),
        DATABASE_URL=_get_env("DATABASE_URL", DATABASE_URL_DEFAULT),
    )


# Глобальный объект настроек. Импортируется другими модулями.
settings = load_settings()