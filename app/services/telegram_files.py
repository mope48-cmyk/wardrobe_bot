"""Скачивание файлов из Telegram Bot API через httpx (с прокси)."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _base() -> str:
    return (settings.TELEGRAM_API_URL or "https://api.telegram.org").rstrip("/")


async def download_photo_by_file_id(file_id: str) -> bytes | None:
    """Скачать фото по file_id (getFile + скачивание)."""
    if not file_id:
        return None

    base = _base()
    token = settings.BOT_TOKEN

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{base}/bot{token}/getFile",
                params={"file_id": file_id},
            )
            r.raise_for_status()
            data = r.json()

        if not data.get("ok"):
            logger.error("getFile вернул ok=false: %s", data)
            return None

        file_path = data["result"]["file_path"]

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(f"{base}/file/bot{token}/{file_path}")
            r.raise_for_status()
            return r.content

    except Exception:
        logger.exception("Не удалось скачать фото file_id=%s", file_id)
        return None
