"""Обработчики команд Telegram-бота."""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

logger = logging.getLogger(__name__)

# Роутер — контейнер для хендлеров. Подключается к Dispatcher в main.py.
router = Router(name="main_router")


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
    """Ответ на любое сообщение, не попавшее в другие хендлеры."""
    await message.answer(
        "Пока я понимаю только команды. Наберите /help, чтобы увидеть список."
    )
