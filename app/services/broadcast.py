"""Рассылка сообщений всем пользователям."""

import asyncio
import hashlib
import logging
import os
import re

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

from app.db.repository import get_all_user_ids

logger = logging.getLogger(__name__)

# Лимит Telegram — 4096 символов на сообщение. Берём с запасом, чтобы чанк
# гарантированно влезал вместе со служебными символами.
MAX_MESSAGE_CHARS = 3800


def _chunk_paragraph(paragraph: str, max_chars: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Разбивает один абзац на фрагменты <= max_chars, сохраняя валидный HTML."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    # Весь файл релиза оформлен как <b>...</b> абзацы — сохраняем обёртку.
    m = re.fullmatch(r"(<b>)(.*)(</b>)", paragraph, flags=re.DOTALL)
    if m:
        open_tag, inner, close_tag = m.groups()
    else:
        open_tag, inner, close_tag = "", paragraph, ""

    sentences = re.split(r"(?<=[.!?])\s+|\n+", inner)
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        candidate = (cur + " " + s).strip() if cur else s
        if len(open_tag + candidate + close_tag) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(open_tag + cur + close_tag)
            cur = s
    if cur:
        chunks.append(open_tag + cur + close_tag)
    return chunks


def _split_message(text: str, max_chars: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Делит текст релиза на сообщения <= max_chars по границам абзацев."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks, cur = [], ""
    for raw in paragraphs:
        p = raw.strip()
        if not p:
            continue
        for frag in _chunk_paragraph(p, max_chars):
            if cur and len(cur) + 1 + len(frag) > max_chars:
                chunks.append(cur)
                cur = frag
            else:
                cur = (cur + "\n\n" + frag) if cur else frag
    if cur:
        chunks.append(cur)
    return chunks


async def send_release_broadcast(
    bot: Bot,
    file_path: str = "release_notes.txt",
    sent_marker: str = "/app/data/release_notes.sent",
) -> None:
    """Читает файл и рассылает сообщение всем пользователям."""
    if not os.path.exists(file_path):
        logger.info("Файл рассылки %s не найден. Пропускаем.", file_path)
        return

    with open(file_path, encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        logger.info("Файл рассылки пуст. Пропускаем.")
        return

    # Хэш текста для проверки, отправляли ли мы его уже
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

    if os.path.exists(sent_marker):
        with open(sent_marker, encoding="utf-8") as f:
            last_hash = f.read().strip()
        if last_hash == text_hash:
            logger.info("Этот релиз уже был успешно разослан ранее. Пропускаем.")
            return

    logger.info("Начинаем рассылку релиза...")
    
    user_ids = await get_all_user_ids()
    logger.info("Найдено %d пользователей для рассылки.", len(user_ids))

    messages = _split_message(text)
    logger.info(
        "Текст релиза разбит на %d сообщений (по <=%d символов).",
        len(messages), MAX_MESSAGE_CHARS,
    )

    success_count = 0
    error_count = 0

    for uid in user_ids:
        ok = True
        for msg_text in messages:
            try:
                await bot.send_message(chat_id=uid, text=msg_text, parse_mode="HTML")
            except TelegramForbiddenError:
                logger.warning("Пользователь %d заблокировал бота.", uid)
                ok = False
                break
            except TelegramBadRequest as e:
                logger.warning("Ошибка отправки пользователю %d: %s", uid, e)
                ok = False
                break
            except Exception:
                logger.exception("Неизвестная ошибка при отправке пользователю %d", uid)
                ok = False
                break

            # Задержка 0.05 сек (лимит Telegram — 30 сообщений в секунду)
            await asyncio.sleep(0.05)

        if ok:
            success_count += 1
        else:
            error_count += 1

    logger.info(
        "Рассылка завершена. Успешно: %d, Ошибок: %d", success_count, error_count
    )

    # Маркер пишем ТОЛЬКО если хотя бы одному пользователю доставили,
    # иначе неудачная попытка навсегда блокирует повторную рассылку.
    if success_count == 0:
        logger.warning(
            "Рассылка не доставлена никому — маркер НЕ сохранён, "
            "будет повторена в следующий запуск."
        )
        return

    try:
        with open(sent_marker, "w", encoding="utf-8") as f:
            f.write(text_hash)
        logger.info("Статус рассылки сохранен в %s", sent_marker)
    except Exception:
        logger.exception("Не удалось сохранить маркер рассылки")