"""Обработчики команд Telegram-бота."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.messages import HumanMessage

from app.graph.builder import build_graph

logger = logging.getLogger(__name__)

router = Router(name="main_router")

# Граф строится один раз при импорте модуля и переиспользуется.
# MemorySaver внутри сохраняет состояние между вызовами.
graph = build_graph()


def _thread_config(chat_id: int) -> dict:
    """Конфиг для LangGraph: одна ветка диалога на один чат."""
    return {"configurable": {"thread_id": str(chat_id)}}


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Обработчик команды /start."""
    user = message.from_user
    logger.info("Команда /start от user_id=%s", user.id if user else "unknown")

    text = (
        f"Привет, {user.first_name if user else 'друг'}!\n\n"
        "Я помогу каталогизировать твой гардероб и подбирать одежду по погоде.\n\n"
        "Доступные команды:\n"
        "/add — добавить вещь\n"
        "/list — показать гардероб\n"
        "/outfit — подобрать комплект по погоде\n"
        "/help — справка"
    )
    await message.answer(text)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Обработчик команды /help."""
    text = (
        "Что я умею:\n\n"
        "• /add — добавление вещи. Пришлите фото, я попробую распознать "
        "категорию и тип, затем задам несколько уточняющих вопросов.\n\n"
        "• /list — показать все вещи в вашем гардеробе.\n\n"
        "• /outfit — подобрать комплект. Я уточню ваше местоположение и повод, "
        "посмотрю прогноз погоды и выберу подходящие вещи.\n\n"
        "Команды /add, /list и /outfit появятся на следующих этапах."
    )
    await message.answer(text)


@router.message()
async def fallback(message: Message) -> None:
    """Любое сообщение, не попавшее в команды, уходит в LangGraph."""
    if not message.text:
        await message.answer("Пока я умею работать только с текстом.")
        return

    user = message.from_user
    config = _thread_config(message.chat.id)

    # Формируем входное состояние. messages — список новых сообщений
    # (add_messages смёржит их с историей).
    input_state = {
        "messages": [HumanMessage(content=message.text)],
        "user_id": user.id if user else 0,
        "chat_id": message.chat.id,
    }

    try:
        result = await graph.ainvoke(input_state, config=config)
    except Exception:
        logger.exception("Ошибка при вызове графа")
        await message.answer("Что-то пошло не так. Попробуйте ещё раз.")
        return

    # Достаём последнее сообщение ассистента
    reply_text = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            reply_text = msg.content
            break

    if reply_text:
        await message.answer(reply_text)
    else:
        await message.answer("Нечего ответить.")