"""Обработчики Telegram-бота."""

import logging

from aiogram import Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from langchain_core.messages import HumanMessage

from app.graph.builder import build_graph

logger = logging.getLogger(__name__)

router = Router(name="main_router")
graph = build_graph()


def _thread_config(chat_id: int) -> dict:
    return {"configurable": {"thread_id": str(chat_id)}}


async def _extract_input(
    message: Message, edit_message_id: int | None = None
) -> dict:
    """Разобрать сообщение Telegram в структурированный ввод.

    Если передан edit_message_id, он попадёт в состояние — это сигнал
    для _send_ai_message отредактировать существующее сообщение вместо
    отправки нового (используется при навигации по спискам).
    """
    input_text = message.text or message.caption or None
    input_photo_file_id = None
    input_photo_bytes = None
    input_location = None

    if message.photo:
        input_photo_file_id = message.photo[-1].file_id
        try:
            buffer = await message.bot.download(input_photo_file_id)
            input_photo_bytes = buffer.read()
            logger.info(
                "Скачано фото file_id=%s размер=%d байт",
                input_photo_file_id, len(input_photo_bytes),
            )
        except Exception:
            logger.exception("Не удалось скачать фото из Telegram")

    if message.location:
        input_location = {
            "lat": message.location.latitude,
            "lon": message.location.longitude,
        }

    return {
        "input_text": input_text,
        "input_photo_file_id": input_photo_file_id,
        "input_photo_bytes": input_photo_bytes,
        "input_location": input_location,
        "edit_message_id": edit_message_id,
    }


def _build_reply_markup(spec):
    """Собрать ReplyKeyboardMarkup из спеки или ReplyKeyboardRemove.

    spec:
      - список рядов (list[list[str]]) → обычная клавиатура
      - строка "remove" → удалить клавиатуру
      - None / отсутствует → ничего не менять
    """
    if spec == "remove":
        return ReplyKeyboardRemove()
    if not spec:
        return None
    rows = [[KeyboardButton(text=t) for t in row] for row in spec]
    return ReplyKeyboardMarkup(
        keyboard=rows,
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def _build_inline_markup(spec: list[list[dict]] | None):
    """Собрать InlineKeyboardMarkup из спеки.

    spec — список рядов, каждый ряд — список dict с полями
    "text" и "callback_data". Например:

        [[{"text": "⬅️ Назад", "callback_data": "list:prev"},
          {"text": "Вперёд ➡️", "callback_data": "list:next"}]]
    """
    if not spec:
        return None
    rows = [
        [
            InlineKeyboardButton(
                text=btn["text"],
                callback_data=btn["callback_data"],
            )
            for btn in row
        ]
        for row in spec
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_ai_message(message: Message, ai_msg) -> None:
    """Отправить одно сообщение ассистента.

    Поддерживает:
    - photo_file_id + caption → answer_photo;
    - edit_message_id + edit_mode:
        * "media"   → edit_message_media (когда меняется фото)
        * "caption" → edit_message_caption (когда фото то же);
    - reply_keyboard / inline_keyboard;
    - пустой content с inline → отправит заглушку "…".
    """
    extra = getattr(ai_msg, "additional_kwargs", None) or {}
    photo_file_id = extra.get("photo_file_id")
    edit_message_id = extra.get("edit_message_id")
    edit_mode = extra.get("edit_mode", "media")
    reply_markup = _build_reply_markup(extra.get("reply_keyboard"))
    inline_markup = _build_inline_markup(extra.get("inline_keyboard"))

    # Inline-клавиатура имеет приоритет: она не «залипает» в чате.
    effective_markup = inline_markup or reply_markup

    # Сценарий 1: редактируем caption существующего сообщения
    # (фото не меняется — используется при подтверждении удаления и отмене)
    if edit_message_id and edit_mode == "caption":
        try:
            await message.bot.edit_message_caption(
                chat_id=message.chat.id,
                message_id=edit_message_id,
                caption=ai_msg.content or "…",
                parse_mode="HTML",
                reply_markup=inline_markup,
            )
            return
        except Exception:
            logger.exception(
                "edit_message_caption не удался, отправляю новое сообщение"
            )

    # Сценарий 2: редактируем media существующего сообщения
    # (навигация по списку: меняется и фото, и caption)
    if edit_message_id and edit_mode == "media" and photo_file_id:
        try:
            await message.bot.edit_message_media(
                chat_id=message.chat.id,
                message_id=edit_message_id,
                media=InputMediaPhoto(
                    media=photo_file_id,
                    caption=ai_msg.content or "…",
                    parse_mode="HTML",
                ),
                reply_markup=inline_markup,
            )
            return
        except Exception:
            logger.exception(
                "edit_message_media не удался, отправляю новое сообщение"
            )

    # Сценарий 3: новое сообщение с фото
    if photo_file_id:
        try:
            await message.answer_photo(
                photo=photo_file_id,
                caption=ai_msg.content or "…",
                reply_markup=effective_markup,
            )
            return
        except Exception:
            logger.exception("Не удалось отправить фото, шлю текстом")

    # Сценарий 4: текстовое сообщение
    await message.answer(ai_msg.content or "…", reply_markup=effective_markup)


@router.message()
async def handle_message(message: Message) -> None:
    """Единый обработчик всех сообщений."""
    user = message.from_user
    parsed = await _extract_input(message, edit_message_id=None)

    if parsed["input_text"]:
        human_content = parsed["input_text"]
    elif parsed["input_photo_file_id"]:
        human_content = "[фото]"
    elif parsed["input_location"]:
        human_content = "[геолокация]"
    else:
        human_content = "[неизвестный ввод]"

    input_state = {
        "messages": [HumanMessage(content=human_content)],
        "user_id": user.id if user else 0,
        "chat_id": message.chat.id,
        **parsed,
    }

    config = _thread_config(message.chat.id)

    try:
        result = await graph.ainvoke(input_state, config=config)
    except Exception:
        logger.exception("Ошибка при вызове графа")
        await message.answer("Что-то пошло не так. Попробуйте ещё раз.")
        return

    all_messages = result["messages"]

    # Ищем последнее сообщение пользователя, чтобы взять только свежие ответы
    last_human_idx = -1
    for i, m in enumerate(all_messages):
        if m.__class__.__name__ == "HumanMessage":
            last_human_idx = i

    new_ai_messages = [
        m for m in all_messages[last_human_idx + 1:]
        if m.__class__.__name__ == "AIMessage"
    ]

    if not new_ai_messages:
        await message.answer("Нечего ответить.")
        return

    for ai_msg in new_ai_messages:
        await _send_ai_message(message, ai_msg)


@router.callback_query()
async def handle_callback(callback: CallbackQuery) -> None:
    """Обработка нажатий на inline-кнопки.

    Callback'и приходят с callback.data, например "list:next".
    Мы передаём их в граф как обычный текстовый ввод — так проще
    переиспользовать логику маршрутизации в route_condition.
    """
    data = callback.data or ""
    user = callback.from_user
    if user is None or callback.message is None:
        await callback.answer()
        return

    # На границах списка — показываем всплывающую подсказку и не идём
    # в граф: незачем перерисовывать то же самое сообщение.
    if data == "list:noop:first":
        await callback.answer("Это первая вещь", show_alert=False)
        return
    if data == "list:noop:last":
        await callback.answer("Это последняя вещь", show_alert=False)
        return

    # Убираем "часики" на кнопке
    await callback.answer()

    # Запоминаем id сообщения, чтобы отредактировать его, а не плодить новые
    edit_message_id = callback.message.message_id

    input_state = {
        "messages": [HumanMessage(content=data)],
        "user_id": user.id,
        "chat_id": callback.message.chat.id,
        "input_text": data,
        "input_photo_file_id": None,
        "input_photo_bytes": None,
        "input_location": None,
        "edit_message_id": edit_message_id,
    }

    config = _thread_config(input_state["chat_id"])
    try:
        result = await graph.ainvoke(input_state, config=config)
    except Exception:
        logger.exception("Ошибка при вызове графа из callback")
        await callback.message.answer("Что-то пошло не так.")
        return

    all_messages = result["messages"]
    last_human_idx = -1
    for i, m in enumerate(all_messages):
        if m.__class__.__name__ == "HumanMessage":
            last_human_idx = i

    new_ai_messages = [
        m for m in all_messages[last_human_idx + 1:]
        if m.__class__.__name__ == "AIMessage"
    ]

    for ai_msg in new_ai_messages:
        await _send_ai_message(callback.message, ai_msg)