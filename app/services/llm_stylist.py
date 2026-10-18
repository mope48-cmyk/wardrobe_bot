"""Генерация комментария от Александра Васильева."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from app.config import settings
from app.db.repository import get_colors

logger = logging.getLogger(__name__)

_llm = None

if settings.GIGACHAT_CREDENTIALS:
    _llm = GigaChat(
        credentials=settings.GIGACHAT_CREDENTIALS,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2-Pro",
        ca_bundle_file="/app/certs/russian_trusted_root_ca.pem",
        temperature=0.1,
        max_tokens=500,
    )
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — комментарии отключены")


async def generate_vasiliev_comment(
    weather: dict, location: str, occasion: str, items: list, max_chars: int = 300
) -> str | None:
    """Сгенерировать остроумный комментарий-рекомендацию в стиле Александра Васильева."""
    if _llm is None or not items:
        return None

    temp = weather.get("feels_like", weather.get("temp", 15))
    desc = weather.get("description", "")
    
    items_text = "\n".join(
        f"- {i.category} / {i.type} (цвет: {', '.join(get_colors(i)) if get_colors(i) else 'неизвестен'}, материал: {i.material or '—'})"
        for i in items
    )

    system_prompt = (
        "Ты — великий историк моды Александр Васильев. Твоя манера речи: театральная, экспрессивная, "
        "с восклицаниями ('Ах, Боже мой!', 'Это просто катастрофа!', 'Восхитительно!', 'Браво!', 'Великолепно!'), "
        "с легкой иронией, французским шармом и упоминанием своего богатого опыта.\n\n"
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать слово 'Шарман' и его производные. Ни при каких обстоятельствах.\n\n"
        "Ты только что лично подобрал этот комплект для пользователя. Твоя задача — презентовать ему "
        "свой выбор как экспертную рекомендацию. ВАЖНО: Ни в коем случае не оценивай выбор пользователя "
        "(он еще ничего не надел). Ты — эксперт, ты советуешь и рекомендуешь. Говори от первого лица: "
        "'Я советую...', 'Я подобрал вам...', 'Этот образ...'.\n\n"
        "В твоем комментарии ОБЯЗАТЕЛЬНО должны быть:\n"
        "1. Место и погода.\n"
        "2. Повод.\n"
        "3. Названия и цвета выбранных вещей (обязательно упомяни цвет!).\n"
        "4. Упоминание известной достопримечательности этого места.\n\n"
        f"ВАЖНОЕ ОГРАНИЧЕНИЕ: Твой комментарий должен быть не длиннее {max_chars} символов! "
        "Пиши кратко и емко, иначе текст не поместится в сообщение."
    )

    user_prompt = (
        f"Место: {location}\n"
        f"Погода: {desc}, {temp:.0f}°C\n"
        f"Повод: {occasion}\n"
        f"Мой подобранный комплект:\n{items_text}\n\n"
        f"Представь свой образ пользователю. Уложись в {max_chars} символов!"
    )

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        text = response.content.strip()
        
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
            
        return text
    except Exception:
        logger.exception("Ошибка генерации комментария Васильева")
        return None