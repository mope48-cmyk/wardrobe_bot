import logging
from datetime import datetime
from html import escape as esc
from zoneinfo import ZoneInfo

from aiogram import Bot
from aiogram.types import BufferedInputFile

from app.db.repository import get_items, update_morning_settings
from app.services.llm_stylist import generate_vasiliev_comment
from app.services.outfit import select_outfit
from app.services.telegram_files import download_photo_by_file_id
from app.services.vogue_cover import build_collage
from app.services.weather import FALLBACK_WEATHER, get_weather

logger = logging.getLogger(__name__)


async def send_morning_push(bot: Bot, chat_id: int, settings) -> None:
    """Сгенерировать и отправить утренний образ."""
    logger.info(f"Начинаем сборку утреннего образа для {chat_id}")

    weather = await get_weather(settings.lat, settings.lon)
    weather_fallback = weather is None
    if weather_fallback:
        logger.warning(
            f"Утренний пуш: погода недоступна для {chat_id}, использую нейтральную"
        )
        weather = FALLBACK_WEATHER

    items = await get_items(chat_id)
    if not items:
        logger.warning(f"Утренний пуш: гардероб пуст для {chat_id}")
        return

    outfits = select_outfit(list(items), weather, settings.occasion)
    if not outfits or (not outfits[0].items and outfits[0].no_candidates):
        logger.warning(f"Утренний пуш: алгоритм не смог собрать образ для {chat_id}")
        return

    best_outfit = outfits[0]
    city_name = settings.city or "вашем городе"

    if weather_fallback:
        weather_line = "⛅ <b>Погода:</b> временно недоступна"
        collage_weather = ""
    else:
        temp = weather["temp"]
        weather_line = f"🌤 <b>Погода:</b> {weather['description']}, {temp:.0f}°C"
        collage_weather = f"{weather['description']}, {temp:.0f}°C"
    comment = await generate_vasiliev_comment(
        weather,
        city_name,
        settings.occasion,
        best_outfit.items,
        relaxed_categories=best_outfit.relaxed_categories,
        max_chars=350,
    )
    if not comment:
        comment = "Браво! Этот образ — отличный выбор на сегодня."

    from app.services.gamification import get_or_create_gam
    gam = await get_or_create_gam(chat_id)
    streak = gam.streak or 0

    caption = (
        f"☀️ <b>Доброе утро!</b>\n\n"
        f"{weather_line}\n"
        f"<b>Повод:</b> {settings.occasion}\n\n"
        f"💬 <i>{esc(comment)}</i>"
    )
    if streak >= 2:
        caption += f"\n\n🔥 Серия: <b>{streak}</b> дн."

    photos = []
    for item in best_outfit.items:
        fid = item.generated_photo_file_id or item.photo_file_id
        img_bytes = await download_photo_by_file_id(fid)
        if img_bytes:
            photos.append(img_bytes)

    if not photos:
        logger.warning(f"Утренний пуш: не удалось скачать фото для {chat_id}")
        return

    try:
        collage_bytes = build_collage(
            photos=photos,
            labels=[i.type for i in best_outfit.items],
            city=settings.city or "",
            weather_str=collage_weather,
            date_str=datetime.now().strftime("%d.%m.%Y"),
        )

        await bot.send_photo(
            chat_id=chat_id,
            photo=BufferedInputFile(collage_bytes, filename="morning.jpg"),
            caption=caption,
            parse_mode="HTML",
        )

        await update_morning_settings(
            chat_id,
            last_sent_date=datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d"),
        )
        logger.info(f"Утренний пуш отправлен для {chat_id}")
    except Exception:
        logger.exception("Ошибка при отправке утреннего пуша")
