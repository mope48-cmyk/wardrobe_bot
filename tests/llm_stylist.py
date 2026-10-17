"""Генерация текстового пояснения к комплекту с помощью LLM."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Инициализируем модели, если ключи заданы
_llm = None

if settings.GROQ_API_KEY and settings.OPENROUTER_API_KEY:
    # Основная модель — Groq (очень быстрая, щедрый бесплатный лимит)
    primary_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=settings.GROQ_API_KEY,
        max_retries=2,
    )
    
    # Резервная модель — OpenRouter (бесплатная Llama 3)
    # Используем ChatOpenAI с кастомным base_url, так как OpenRouter
    # полностью совместим с API OpenAI.
    fallback_llm = ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct:free",
        temperature=0.3,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_retries=2,
    )
    
    # Если Groq упадёт с ошибкой лимита (или любой другой) — запрос уйдёт в OpenRouter
    _llm = primary_llm.with_fallbacks([fallback_llm])

elif settings.GROQ_API_KEY:
    _llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=settings.GROQ_API_KEY
    )
elif settings.OPENROUTER_API_KEY:
    _llm = ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct:free",
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


async def generate_outfit_explanation(
    weather: dict, occasion: str, items: list
) -> str | None:
    """Сгенерировать короткое объяснение, почему этот комплект подходит.
    
    Возвращает строку или None, если LLM недоступна/упала.
    """
    if _llm is None:
        logger.warning("LLM не настроена (нет ключей)")
        return None

    # Берем ощущаемую температуру, если есть
    temp = weather.get("feels_like", weather.get("temp", 15))
    desc = weather.get("description", "")
    wind = weather.get("wind_speed", 0)

    # Формируем список вещей для промпта
    items_text = "\n".join(
        f"- {i.category} / {i.type} (материал: {i.material or 'неизвестен'})"
        for i in items
    )

    system_prompt = (
        "Ты — профессиональный, но дружелюбный стилист. "
        "Твоя задача — коротко (2-3 предложения) объяснить пользователю, "
        "почему подобранный комплект одежды идеально подходит под текущую погоду и повод. "
        "Не придумывай вещи, которых нет в списке. Опирайся только на факты о погоде и вещах. "
        "Не используй сложные термины, пиши простым языком."
    )

    user_prompt = (
        f"Погода: {desc}, ощущается как {temp:.0f}°C, ветер {wind:.0f} м/с.\n"
        f"Повод: {occasion}.\n"
        f"Подобранные вещи:\n{items_text}\n\n"
        "Напиши короткое объяснение для пользователя."
    )

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content.strip()
    except Exception:
        logger.exception("Ошибка при генерации текста LLM")
        return None
