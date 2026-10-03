"""Точка входа Telegram-бота."""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.bot.handlers import router
from app.config import settings


def setup_logging() -> None:
    """Настроить логирование в консоль."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def main() -> None:
    """Создать бота, подключить роутеры и запустить polling."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # DefaultBotProperties задаёт parse_mode по умолчанию для всех сообщений.
    # ParseMode.HTML позволяет использовать <b>, <i>, <code> в текстах.
    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Бот запускается. Нажмите Ctrl+C для остановки.")

    # Удаляем старые обновления (накопившиеся, пока бот был выключен),
    # чтобы не обрабатывать их после запуска.
    await bot.delete_webhook(drop_pending_updates=True)

    # start_polling — циклически опрашивает Telegram на новые сообщения.
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nБот остановлен.")
