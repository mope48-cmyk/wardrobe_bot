"""Анализ одежды по фото через мультимодальную GigaChat-2-Pro."""

import base64
import io
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

MAX_SIDE = 1024

_llm = None
if settings.GIGACHAT_CREDENTIALS:
    _llm = GigaChat(
        credentials=settings.GIGACHAT_CREDENTIALS,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2-Pro",
        ca_bundle_file="/app/certs/russian_trusted_root_ca.pem",
        temperature=0.1,
        max_tokens=500,
        auto_upload_attachments=True,
    )
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — анализ одежды отключен")


def _prepare_image_b64(photo_bytes: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            ratio = MAX_SIDE / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.exception("Не удалось подготовить изображение для GigaChat")
        return None


def _extract_json(text: str) -> dict | None:
    """Вытащить JSON из ответа LLM (на случай markdown-оберток)."""
    try:
        # Ищем все от { до }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group(0))
    except Exception:
        logger.exception("Не удалось распарсить JSON: %s", text[:100])
        return None


async def analyze_clothing_with_gigachat(image_bytes: bytes) -> dict | None:
    """Определить характеристики вещи по фото.

    Возвращает dict с полями:
    category, type, colors, material, warmth_level,
    waterproof, formal_level, season, layer.
    """
    if _llm is None:
        return None

    b64 = _prepare_image_b64(image_bytes)
    if not b64:
        return None

    system_prompt = (
        "Ты — экспертный анализатор одежды. Оцени вещь на фото и верни СТРОГО JSON без комментариев.\n"
        "Поля JSON:\n"
        "- category: 'верх', 'низ', 'обувь' или 'аксессуар'\n"
        "- type: тип вещи (например, 'футболка', 'джинсы', 'кожаная куртка')\n"
        "- colors: список из 1-3 основных цветов на русском (например, ['черный', 'белый'])\n"
        "- material: материал на русском (хлопок, кожа, джинса, шерсть и т.д.)\n"
        "- warmth_level: число от 1 (очень легко) до 5 (очень тепло)\n"
        "- waterproof: true/false (может ли промокнуть)\n"
        "- formal_level: 'casual', 'business' или 'sport'\n"
        "- season: 'лето', 'демисезон', 'зима' или 'универсальная'\n"
        "- layer: 'base', 'mid', 'outer' или 'none' (для верха)\n\n"
        "Верни только JSON. Не пиши ```json."
    )

    user_content = [
        {"type": "text", "text": "Проанализируй эту вещь."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])

        text = response.content.strip()
        data = _extract_json(text)
        if not data:
            return None

        # Базовая валидация типов
        valid_categories = {"верх", "низ", "обувь", "аксессуар"}
        valid_formal = {"casual", "business", "sport"}
        valid_seasons = {"лето", "демисезон", "зима", "универсальная"}
        valid_layers = {"base", "mid", "outer", "none"}

        if data.get("category") not in valid_categories:
            return None
        if data.get("formal_level") not in valid_formal:
            data["formal_level"] = "casual"
        if data.get("season") not in valid_seasons:
            data["season"] = "универсальная"
        if data.get("layer") not in valid_layers:
            data["layer"] = "none" if data.get("category") != "верх" else "base"

        # Ограничиваем цвета
        colors = data.get("colors") or []
        if not isinstance(colors, list):
            colors = [str(colors)]
        data["colors"] = [str(c).strip().lower() for c in colors[:3]]

        data["warmth_level"] = max(1, min(5, int(data.get("warmth_level", 3))))

        return data
    except Exception:
        logger.exception("Ошибка GigaChat при анализе одежды")
        return None
