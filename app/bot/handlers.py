"""Обработчики Telegram-бота."""

import logging

from aiogram import Router
from aiogram.types import Message
from langchain_core.messages import HumanMessage

from app.graph.builder import build_graph

logger = logging.getLogger(__name__)

router = Router(name="main_router")
graph = build_graph()


def _thread_config(chat_id: int) -> dict:
    return {"configurable": {"thread_id": str(chat_id)}}


async def _extract_input(message: Message) -> dict:
    """Разобрать сообщение в структурированный ввод.

    Если есть фото — скачиваем его в память, чтобы передать в граф.
    """
    input_text = message.text or message.caption or None
    input_photo_file_id = None
    input_photo_bytes = None
    input_location = None

    if message.photo:
        input_photo_file_id = message.photo[-1].file_id
        try:
            # download() возвращает io.BytesIO
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

    reply_text = ""
    for msg in reversed(result["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            reply_text = msg.content
            break

    if reply_text:
        await message.answer(reply_text)
    else:
        await message.answer("Нечего ответить.")