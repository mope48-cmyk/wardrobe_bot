"""Экспорт реальных фото вещей из БД в датасет для оценки.

Скачивает фото по photo_file_id (Telegram Bot API) в выходную папку
и пишет рядом manifest.csv с метками из БД (для tools/eval_recognition.py).

Запуск на сервере (внутри контейнера):
    docker compose exec bot python tools/export_real_dataset.py /app/dataset_real
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

# Корень проекта (родитель tools/) — чтобы работал `import app.*`
# при запуске как `python tools/export_real_dataset.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("export_real_dataset")


def primary_color(raw: str) -> str:
    try:
        arr = json.loads(raw or "[]")
        return str(arr[0]).strip() if arr else ""
    except Exception:
        return ""


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir", type=Path, nargs="?", default=Path("/app/dataset_real")
    )
    args = parser.parse_args()

    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties
    from aiogram.client.session.aiohttp import AiohttpSession
    from aiogram.client.telegram import TelegramAPIServer
    from aiogram.enums import ParseMode
    from sqlalchemy import select

    from app.config import settings
    from app.db.database import SessionLocal
    from app.db.models import WardrobeItem

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    session_arg = None
    if settings.TELEGRAM_API_URL:
        logger.info("Используем прокси: %s", settings.TELEGRAM_API_URL)
        session_arg = AiohttpSession(
            api=TelegramAPIServer.from_base(settings.TELEGRAM_API_URL)
        )
    bot = Bot(
        token=settings.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=session_arg,
    )

    try:
        async with SessionLocal() as s:
            items = (await s.execute(select(WardrobeItem))).scalars().all()

        rows = []
        downloaded = skipped = failed = 0
        for it in items:
            target = out / f"{it.id:04d}.jpg"
            if target.exists():
                skipped += 1
            else:
                try:
                    file_obj = await bot.get_file(it.photo_file_id)
                    await bot.download_file(file_obj.file_path, destination=target)
                    downloaded += 1
                except Exception:
                    logger.exception("Не удалось скачать вещь id=%s", it.id)
                    failed += 1
                    continue
            rows.append({
                "file": target.name,
                "category": (it.category or "").strip().lower(),
                "type": (it.type or "").strip(),
                "color": primary_color(it.colors),
            })

        manifest = out / "manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "category", "type", "color"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Готово: скачано=%d, уже было=%d, ошибок=%d, всего=%d",
            downloaded, skipped, failed, len(items),
        )
        logger.info("Манифест: %s", manifest)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
