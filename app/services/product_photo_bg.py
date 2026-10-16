"""Фоновая генерация фото для каталога.

Запускается из handlers после нажатия кнопки.
Пока идёт генерация, мигает "отправляет фото" в шапке чата,
когда готово — приходит уведомление + карточка обновляется.
"""

import asyncio
import logging

from aiogram import Bot
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


async def _chat_action_loop(
    bot: Bot, chat_id: int, stop: asyncio.Event
) -> None:
    """Пока генерация идёт — каждые 4 сек посылаем 'отправляет фото'."""
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
    """Inline-клавиатура карточки, когда фото из каталога уже есть."""
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
    """Собрать caption карточки с фото из каталога."""
    item = await get_item(item_id)
    if item is None:
        return "📋 Вещь не найдена"

    colors_str = ", ".join(get_colors(item)) or "—"
    material_line = f"материал: {item.material}\n" if item.material else ""
    waterproof_mark = "💧 " if item.waterproof else ""

    return (
        f"📋 Вещь <b>{card_index + 1}</b> из <b>{card_total}</b>\n\n"
        f"<b>{item.category} / {item.type}</b>\n"
        f"цвета: {colors_str}\n"
        f"{material_line}"
        f"{waterproof_mark}тепло: {item.warmth_level}/5, "
        f"стиль: {item.formal_level}, сезон: {item.season}\n\n"
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
    """Сгенерировать фото для каталога в фоне.

    Не блокирует обработку других сообщений пользователя.
    """
    stop = asyncio.Event()
    action_task = asyncio.create_task(
        _chat_action_loop(bot, chat_id, stop)
    )

    try:
        # 1. Скачиваем оригинал
        image_bytes = await download_photo_by_file_id(photo_file_id)
        if not image_bytes:
            await bot.send_message(
                chat_id, "❌ Не удалось скачать фото вещи."
            )
            return

        # 2. Генерация — синхронная, тяжёлая, в отдельный поток
        product_bytes = await asyncio.to_thread(
            make_product_photo, image_bytes, category
        )

        # 3. Загружаем в Telegram — это и есть уведомление
        caption = (
            f"✅ Фото для каталога готово\n"
            f"• {category} / {item_type}"
        )
        sent = await bot.send_photo(
            chat_id=chat_id,
            photo=BufferedInputFile(product_bytes, filename="catalog.jpg"),
            caption=caption,
        )
        new_file_id = sent.photo[-1].file_id if sent.photo else None

        if not new_file_id:
            logger.error("Telegram не вернул file_id после send_photo")
            return

        # 4. Сохраняем file_id в БД
        ok = await update_item_field(
            item_id, user_id, "generated_photo_file_id", new_file_id
        )
        if not ok:
            logger.error(
                "update_item_field вернул False для item_id=%s", item_id
            )
            return

        logger.info(
            "Фото каталога сохранено: item_id=%s file_id=%s",
            item_id, new_file_id,
        )

        # 5. Пытаемся обновить карточку, если она ещё существует
        if card_message_id:
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
                logger.info(
                    "Карточка обновлена после генерации: message_id=%s",
                    card_message_id,
                )
            except Exception:
                # Сообщение могло быть удалено или перезаписано — норм
                logger.info(
                    "Карточку не удалось обновить (message_id=%s): "
                    "пользователь ушёл или сообщение удалено",
                    card_message_id,
                )

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
        stop.set()
        action_task.cancel()
