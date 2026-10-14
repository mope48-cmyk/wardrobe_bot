from app.services.vision import warmup
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ParseMode

from app.bot.handlers import router
from app.config import settings
from app.db.database import init_db


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    await init_db()

    # Прогрев моделей в фоне, чтобы первый запрос не тормозил
    asyncio.create_task(asyncio.to_thread(warmup))

    # Если задан TELEGRAM_API_URL — идём в Telegram через прокси.
    # Иначе — напрямую в api.telegram.org.
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
    dp.include_router(router)

    logger.info("Бот запускается. Нажмите Ctrl+C для остановки.")

    await bot.delete_webhook(drop_pending_updates=True)

    # Прогреваем модели (SigLIP + Florence-2) до старта polling,
    # чтобы первый запрос с фото не тормозил.
    # warmup() синхронный и блокирующий — выносим в отдельный поток,
    # чтобы не блокировать event loop.
    logger.info("Прогрев моделей...")
    await asyncio.to_thread(warmup)
    logger.info("Прогрев завершён, запускаем polling")

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nБот остановлен.")