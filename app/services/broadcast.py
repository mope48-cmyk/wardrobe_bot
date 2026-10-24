"""Рассылка сообщений всем пользователям."""

import asyncio
import hashlib
import logging
import os

from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest

from app.db.repository import get_all_user_ids

logger = logging.getLogger(__name__)

async def send_release_broadcast(bot: Bot, file_path: str = "release_notes.txt") -> None:
    """Читает файл и рассылает сообщение всем пользователям."""
    if not os.path.exists(file_path):
        logger.info("Файл рассылки %s не найден. Пропускаем.", file_path)
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        logger.info("Файл рассылки пуст. Пропускаем.")
        return

    # Хэш текста для проверки, отправляли ли мы его уже
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # Файл-маркер в папке data (куда у нас есть права на запись)
    sent_marker = "/app/data/release_notes.sent"

    if os.path.exists(sent_marker):
        with open(sent_marker, "r", encoding="utf-8") as f:
            last_hash = f.read().strip()
        if last_hash == text_hash:
            logger.info("Этот релиз уже был успешно разослан ранее. Пропускаем.")
            return

    logger.info("Начинаем рассылку релиза...")
    
    user_ids = await get_all_user_ids()
    logger.info("Найдено %d пользователей для рассылки.", len(user_ids))

    success_count = 0
    error_count = 0

    for uid in user_ids:
        try:
            await bot.send_message(chat_id=uid, text=text, parse_mode="HTML")
            success_count += 1
        except TelegramForbiddenError:
            logger.warning("Пользователь %d заблокировал бота.", uid)
            error_count += 1
        except TelegramBadRequest as e:
            logger.warning("Ошибка отправки пользователю %d: %s", uid, e)
            error_count += 1
        except Exception:
            logger.exception("Неизвестная ошибка при отправке пользователю %d", uid)
            error_count += 1
        
        # Задержка 0.05 сек (лимит Telegram — 30 сообщений в секунду)
        await asyncio.sleep(0.05)

    logger.info("Рассылка завершена. Успешно: %d, Ошибок: %d", success_count, error_count)

    # Сохраняем хэш отправленного сообщения в маркер
    try:
        with open(sent_marker, "w", encoding="utf-8") as f:
            f.write(text_hash)
        logger.info("Статус рассылки сохранен в /app/data/release_notes.sent")
    except Exception:
        logger.exception("Не удалось сохранить маркер рассылки")