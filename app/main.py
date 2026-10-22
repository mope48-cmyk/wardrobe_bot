import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ParseMode

from app.bot.handlers import router
from app.bot.middlewares import AlbumMiddleware
from app.config import settings
from app.db.database import init_db
from app.services.vision import warmup


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    await init_db()

    # Если задан TELEGRAM_API_URL — идём в Telegram через прокси.
    if settings.TELEGRAM_API_URL:
        logger.info("Используем прокси: %s", settings.TELEGRAM_API_URL)
        session = AiohttpSession(
            api=TelegramAPIServer.from_base(settings.TELEGRAM_API_URL)
        )
    else:
        logger.info("Прокси не задан, работаем напрямую с api.telegram.org")
        session = None

    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=session,
    )

    dp = Dispatcher()
    dp.message.middleware(AlbumMiddleware())  # <-- Добавить эту строку
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)

    # Прогреваем модели до старта polling, чтобы первое фото не тормозило.
    # warmup() синхронный и блокирующий — выносим в отдельный поток.
    logger.info("Прогрев моделей...")
    await asyncio.to_thread(warmup)
    logger.info("Прогрев завершён, запускаем polling")

    logger.info("Бот запускается. Нажмите Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nБот остановлен.")