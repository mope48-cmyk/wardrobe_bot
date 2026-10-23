"""«Приговор от Васильева» — оценка образа пользователя по фото.

Использует мультимодальную GigaChat-2-Pro через base64 data-url +
auto_upload_attachments (langchain-gigachat 0.5.x).
"""

import base64
import io
import logging

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
        temperature=0.3,
        max_tokens=2000,                 # ← было 700
        auto_upload_attachments=True,  # для мультимодальности в 0.5.x
    )
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — приговоры отключены")


def _prepare_image_b64(photo_bytes: bytes) -> str | None:
    """Сжать фото до MAX_SIDE по длинной стороне → base64 JPEG."""
    try:
        img = Image.open(io.BytesIO(photo_bytes))
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            ratio = MAX_SIDE / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        logger.exception("Не удалось подготовить изображение для вердикта")
        return None


async def generate_verdict_vasiliev(photo_bytes: bytes) -> str | None:
    """Сгенерировать «приговор» образу по фото в стиле Васильева.

    Длина не ограничивается — разбиением на сообщения занимается nodes.py.
    """
    if _llm is None:
        return None

    b64 = _prepare_image_b64(photo_bytes)
    if not b64:
        return None

    system_prompt = (
        "Ты — великий историк моды Александр Васильев. Твоя манера речи: "
        "театральная, экспрессивная, с лёгкой иронией и французским шармом.\n\n"
        "КАТЕГОРИЧЕСКИЕ ЗАПРЕТЫ:\n"
        "1. ЗАПРЕЩЕНО слово 'Шарман'.\n"
        "2. ЗАПРЕЩЕНО начинать предложения со слова 'Ах'. Используй другие "
        "восклицания: 'Боже мой!', 'Восхитительно!', 'Это катастрофа!', "
        "'Позор!', 'Браво!', 'Какая прелесть!'.\n\n"
        "Пользователь прислал фото себя в полный рост. Вынеси «приговор» "
        "его образу. Структура:\n"
        "1. Общее впечатление — одним предложением.\n"
        "2. Что удалось — конкретно: цвет, силуэт, обувь, аксессуары, посадка.\n"
        "3. Что не так — конструктивно, но в твоём стиле (можно жёстко).\n"
        "4. Одна конкретная рекомендация, что исправить.\n\n"
        "Говори от первого лица: 'Я вижу...', 'Мне открывается...', "
        "'Я бы посоветовал...'. Ссылайся на то, что РЕАЛЬНО видно на фото. "
        "Не выдумывай вещи, которых нет.\n\n"
        "Излагай мысль развёрнуто — не бойся быть подробным, но избегай воды. "
        "Если рекомендаций несколько — перечисли их по пунктам."
    )

    user_content = [
        {"type": "text", "text": "Оцени этот образ и вынеси приговор."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        return response.content.strip()   # ← без обрезки
    except Exception:
        logger.exception("Ошибка генерации приговора от Васильева")
        return None
