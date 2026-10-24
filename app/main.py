import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.bot.handlers import router
from app.bot.middlewares import AlbumMiddleware
from app.config import settings
from app.db.database import init_db
from app.services.broadcast import send_release_broadcast
from app.services.vision import warmup


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# Инициализируем планировщик (пока без запуска)
scheduler = AsyncIOScheduler(timezone="Europe/Moscow")


async def scheduled_broadcast(bot):
    """Обертка для отправки, чтобы ловить ошибки в фоне."""
    try:
        logging.info("⏰ Запуск ежедневной проверки релиз-ноутс...")
        await send_release_broadcast(bot, file_path="release_notes.txt")
    except Exception:
        logging.exception("Ошибка в ежедневной рассылке релиза")


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

    # НАСТРОЙКА РАССЫЛКИ
    # Добавляем задачу: каждый день в 10:00
    scheduler.add_job(
        scheduled_broadcast,
        trigger=CronTrigger(hour=10, minute=0, timezone="Europe/Moscow"),
        args=[bot],
        id="daily_release_broadcast",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Планировщик запущен. Ежедневная рассылка в 10:00 МСК.")

    logger.info("Бот запускается. Нажмите Ctrl+C для остановки.")
    try:
        await dp.start_polling(bot)
    finally:
        # При выключении бота корректно останавливаем планировщик
        scheduler.shutdown()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nБот остановлен.")