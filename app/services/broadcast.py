"""Рассылка сообщений всем пользователям."""

import asyncio
import logging
import os

from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest

from app.db.repository import get_all_user_ids

logger = logging.getLogger(__name__)

async def send_release_broadcast(bot: Bot, file_path: str = "release_notes.txt") -> None:
    """Читает файл и рассылает сообщение всем пользователям."""
    # 1. Проверяем, существует ли файл
    if not os.path.exists(file_path):
        logger.info("Файл рассылки %s не найден. Пропускаем.", file_path)
        return

    # 2. Читаем текст
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        logger.info("Файл рассылки пуст. Пропускаем.")
        return

    logger.info("Начинаем рассылку релиза...")
    
    # 3. Получаем ID всех пользователей
    user_ids = await get_all_user_ids()
    logger.info("Найдено %d пользователей для рассылки.", len(user_ids))

    success_count = 0
    error_count = 0

    # 4. Рассылаем
    for uid in user_ids:
        try:
            await bot.send_message(chat_id=uid, text=text, parse_mode="HTML")
            success_count += 1
        except TelegramForbiddenError:
            # Пользователь заблокировал бота (это нормально)
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

    # 5. Переименовываем файл, чтобы не отправить его снова при перезапуске
    try:
        os.rename(file_path, file_path + ".sent")
        logger.info("Файл рассылки переименован в %s.sent", file_path)
    except Exception:
        logger.exception("Не удалось переименовать файл рассылки")
