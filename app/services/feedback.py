"""Обратная связь на образы (кнопки «👍 отлично» / «👎 не то»)."""

import json
import logging

from app.db.database import SessionLocal
from app.db.models import OutfitFeedback

logger = logging.getLogger(__name__)


async def record_outfit_feedback(
    user_id: int,
    verdict: str,
    occasion: str = "",
    weather_main: str = "",
    weather_temp: float = 0.0,
    outfit_types: list[str] | None = None,
) -> None:
    """Сохранить оценку образа."""
    if verdict not in ("like", "dislike"):
        logger.warning("Неизвестный вердикт фидбек-а: %s", verdict)
        return

    async with SessionLocal() as session:
        feedback = OutfitFeedback(
            user_id=user_id,
            verdict=verdict,
            occasion=occasion or "",
            weather_main=weather_main or "",
            weather_temp=weather_temp or 0.0,
            outfit_types=json.dumps(outfit_types or [], ensure_ascii=False),
        )
        session.add(feedback)
        await session.commit()


async def get_feedback_stats() -> dict:
    """Сводка по оценкам: всего, лайков, дизлайков, рейтинг (для админ-панели)."""
    async with SessionLocal() as session:
        from sqlalchemy import func, select

        total = await session.scalar(select(func.count()).select_from(OutfitFeedback))
        likes = await session.scalar(
            select(func.count()).where(OutfitFeedback.verdict == "like")
        )
        dislikes = await session.scalar(
            select(func.count()).where(OutfitFeedback.verdict == "dislike")
        )
    total = total or 0
    likes = likes or 0
    dislikes = dislikes or 0
    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "match_rate": round(likes / total, 3) if total else 0.0,
    }
