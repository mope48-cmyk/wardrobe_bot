"""Фоновая генерация фото для каталога.

Пока идёт — мигает "отправляет фото" в шапке чата.
Когда готово:
  - сохраняем file_id в БД;
  - если пользователь всё ещё смотрит на эту вещь — обновляем карточку;
  - всегда шлём текстовое уведомление.
"""

import asyncio
import logging

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
)

from app.db.repository import get_colors, get_item, update_item_field
from app.services.product_photo import make_product_photo
from app.services.telegram_files import download_photo_by_file_id

logger = logging.getLogger(__name__)

COLOR_EMOJI: dict[str, str] = {
    "красный": "🔴", "бордовый": "🔴", "розовый": "🩷",
    "оранжевый": "🟠", "терракотовый": "🟠",
    "жёлтый": "🟡", "горчичный": "🟡",
    "зелёный": "🟢", "оливковый": "🟢", "хаки": "🟢", "мятный": "🟢",
    "голубой": "🔵", "бирюзовый": "🔵",
    "синий": "🔵", "тёмно-синий": "🔵",
    "фиолетовый": "🟣", "сиреневый": "🟣", "лиловый": "🟣",
    "коричневый": "🟤", "тёмно-коричневый": "🟤",
    "белый": "⚪", "кремовый": "⚪", "бежевый": "⚪",
    "чёрный": "⚫",
    "серый": "🩶", "тёмно-серый": "🩶", "светло-серый": "🩶",
}


def _format_colors(item_colors: list[str]) -> str:
    if not item_colors:
        return "—"
    return ", ".join(
        f"{COLOR_EMOJI.get(c.lower(), '🎨')} {c}" for c in item_colors
    )


# In-memory: item_id, для которых идёт генерация сейчас.
_generating: set[int] = set()

# In-memory: какой item_id сейчас показывается в карточке у chat_id.
_current_card: dict[int, int] = {}


def is_generating(item_id: int) -> bool:
    return item_id in _generating


def mark_generating(item_id: int) -> None:
    _generating.add(item_id)


def set_current_card(chat_id: int, item_id: int) -> None:
    _current_card[chat_id] = item_id


async def _chat_action_loop(
    bot: Bot, chat_id: int, stop: asyncio.Event
) -> None:
    """Каждые 4 сек посылаем 'отправляет фото', пока генерация идёт."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, "upload_photo")
        except Exception:
            logger.exception("send_chat_action failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue


def _card_keyboard(index: int, total: int) -> InlineKeyboardMarkup:
    """Клавиатура карточки (используется только при обновлении самой карточки в шаге 5)."""
    prev_cb = "list:prev" if index > 0 else "list:noop:first"
    next_cb = "list:next" if index < total - 1 else "list:noop:last"

    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="✏️ Редактировать", callback_data="list:edit"
            ),
            InlineKeyboardButton(
                text="🗑 Удалить вещь", callback_data="list:delete"
            ),
        ],
        [
            InlineKeyboardButton(
                text="📷 Оригинал", callback_data="list:toggle_photo"
            ),
            InlineKeyboardButton(
                text="🗑 Удалить фото каталога",
                callback_data="list:delete_product_photo",
            ),
        ],
        [
            InlineKeyboardButton(
                text="⬅️ Назад", callback_data=prev_cb
            ),
            InlineKeyboardButton(
                text="Вперёд ➡️", callback_data=next_cb
            ),
        ],
    ])


async def _build_card_caption(
    item_id: int, card_index: int, card_total: int
) -> str:
    """Собрать caption карточки вещи для обновления и уведомления.
    
    Компактный формат: цвет, материал и слой в одну строку.
    """
    item = await get_item(item_id)
    if item is None:
        return ""

    parts = []
    
    item_colors = get_colors(item)
    if item_colors:
        parts.append(f"цвета: {_format_colors(item_colors)}")
        
    if item.material:
        parts.append(f"материал: {item.material}")
        
    layer_map = {
        "base": "базовый",
        "mid": "средний",
        "outer": "верхний",
        "none": "нет",
    }
    layer_ru = layer_map.get(getattr(item, "layer", "base"), "базовый")
    parts.append(f"слой: {layer_ru}")
    
    attrs_str = " | ".join(parts)
    waterproof_mark = "💧 " if item.waterproof else ""

    return (
        f"📋 Вещь <b>{card_index + 1}</b> из <b>{card_total}</b>\n\n"
        f"<b>{item.category} / {item.type}</b>\n"
        f"{attrs_str}\n"
        f"{waterproof_mark}тепло: {item.warmth_level}/5 | "
        f"стиль: {item.formal_level} | сезон: {item.season}\n\n"
        f"📸 <i>фото из каталога</i>"
    )


async def generate_product_photo_bg(
    *,
    bot: Bot,
    chat_id: int,
    card_message_id: int | None,
    item_id: int,
    user_id: int,
    photo_file_id: str,
    category: str,
    item_type: str,
    card_index: int,
    card_total: int,
) -> None:
    """Сгенерировать фото для каталога в фоне."""
    mark_generating(item_id)
    stop = asyncio.Event()
    action_task = asyncio.create_task(
        _chat_action_loop(bot, chat_id, stop)
    )

    try:
        # 1. Скачиваем оригинал
        image_bytes = await download_photo_by_file_id(photo_file_id)
        if not image_bytes:
            await bot.send_message(chat_id, "❌ Не удалось скачать фото вещи.")
            return

        # 2. Генерация — синхронная, тяжёлая → в отдельный поток
        product_bytes = await asyncio.to_thread(
            make_product_photo, image_bytes, category
        )

        # 3. ВРЕМЕННО загружаем в Telegram, чтобы получить file_id.
        sent = await bot.send_photo(
            chat_id=chat_id,
            photo=BufferedInputFile(product_bytes, filename="catalog.jpg"),
            caption="⏳ Загрузка фото…",
        )
        new_file_id = sent.photo[-1].file_id if sent.photo else None

        try:
            await bot.delete_message(
                chat_id=chat_id, message_id=sent.message_id
            )
        except Exception:
            logger.exception("Не удалось удалить временное сообщение с фото")

        if not new_file_id:
            logger.error("Telegram не вернул file_id после send_photo")
            return

        # 4. Сохраняем file_id в БД
        ok = await update_item_field(
            item_id, user_id, "generated_photo_file_id", new_file_id
        )
        if not ok:
            logger.warning(
                "update_item_field вернул False: item_id=%s "
                "(вероятно, вещь удалена)", item_id,
            )
            return

        logger.info(
            "Фото каталога сохранено: item_id=%s file_id=%s",
            item_id, new_file_id,
        )

        # 5. Обновляем карточку, только если пользователь всё ещё на ней
        if card_message_id and _current_card.get(chat_id) == item_id:
            try:
                card_caption = await _build_card_caption(
                    item_id, card_index, card_total
                )
                await bot.edit_message_media(
                    chat_id=chat_id,
                    message_id=card_message_id,
                    media=InputMediaPhoto(
                        media=new_file_id,
                        caption=card_caption,
                        parse_mode="HTML",
                    ),
                    reply_markup=_card_keyboard(card_index, card_total),
                )
                logger.info("Карточка обновлена: message_id=%s", card_message_id)
            except TelegramBadRequest as e:
                logger.info(
                    "Карточку не обновили (message_id=%s): %s",
                    card_message_id, e,
                )
            except Exception:
                logger.exception("Ошибка обновления карточки")

        # 6. ТЕКСТОВОЕ уведомление без фото и кнопок
        try:
            card_caption = await _build_card_caption(
                item_id, card_index, card_total
            )
            notification_text = (
                f"✅ <b>Фото для каталога вещи «{item_type}» готово!</b>\n\n"
                f"{card_caption}"
            )
            await bot.send_message(
                chat_id=chat_id,
                text=notification_text,
                # reply_markup удален, чтобы не было кнопок
            )
        except Exception:
            logger.exception("Не удалось отправить текстовое уведомление")

    except Exception:
        logger.exception("Ошибка фоновой генерации фото")
        try:
            await bot.send_message(
                chat_id,
                "❌ Не удалось подготовить фото для каталога. "
                "Попробуйте позже.",
            )
        except Exception:
            logger.exception("Не удалось отправить сообщение об ошибке")
    finally:
        _generating.discard(item_id)
        stop.set()
        action_task.cancel()