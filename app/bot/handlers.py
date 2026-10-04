"""Обработчики Telegram-бота."""

import logging

from aiogram import Router
from aiogram.types import (
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


async def _extract_input(message: Message) -> dict:
    """Разобрать сообщение Telegram в структурированный ввод."""
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


async def _send_ai_message(message: Message, ai_msg) -> None:
    """Отправить одно сообщение ассистента: фото или текст."""
    photo_file_id = None
    reply_markup = None
    if getattr(ai_msg, "additional_kwargs", None):
        photo_file_id = ai_msg.additional_kwargs.get("photo_file_id")
        reply_markup = _build_reply_markup(
            ai_msg.additional_kwargs.get("reply_keyboard")
        )

    if photo_file_id:
        try:
            await message.answer_photo(
                photo=photo_file_id,
                caption=ai_msg.content,
                reply_markup=reply_markup,
            )
            return
        except Exception:
            logger.exception("Не удалось отправить фото, шлю текстом")
    await message.answer(ai_msg.content, reply_markup=reply_markup)


@router.message()
async def handle_message(message: Message) -> None:
    """Единый обработчик всех сообщений."""
    user = message.from_user
    parsed = await _extract_input(message)

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