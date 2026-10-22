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
        f"- {i.category} / {i.type} "
        f"(цвет: {', '.join(get_colors(i)) if get_colors(i) else 'неизвестен'}, "
        f"материал: {i.material or '—'}, тепло: {i.warmth_level}/5)"
        for i in items
    )

    # Сумма тепла — чтобы LLM могла вычислить правило пункта 5
    total_warmth = sum(i.warmth_level for i in items)
    is_cold_catastrophe = temp < 10 and total_warmth < 8

    system_prompt = (
        "Ты — великий историк моды Александр Васильев. Твоя манера речи: театральная, экспрессивная, "
        "с восклицаниями ('Ах, Боже мой!', 'Это просто катастрофа!', 'Восхитительно!', 'Браво!'), "
        "с легкой иронией и французским шармом.\n\n"
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО слово 'Шарман'.\n\n"
        "Ты только что лично подобрал этот комплект. Говори только от первого лица: 'Я подобрал вам...', 'Мне пришлось выбрать...'.\n\n"
        "В твоем комментарии ОБЯЗАТЕЛЬНО должны быть:\n"
        "1. Место и погода.\n"
        "2. Повод.\n"
        "3. Названия и цвета выбранных вещей (упомяни цвет!).\n"
        "4. Упоминание известной достопримечательности этого места.\n"
        "5. ЕСЛИ НА УЛИЦЕ ХОЛОДНО (ниже +10°C), А СУММА ТЕПЛА ВЕЩЕЙ МЕНЬШЕ 8 — назови это катастрофой и скажи, что человек замерзнет.\n"
        "6. ЕСЛИ ПОВОД 'СПОРТ', А В СПИСКЕ ЕСТЬ ВЕЩИ СТИЛЯ 'BUSINESS' ИЛИ 'CASUAL' (например, туфли, брюки, рубашки) — ты ОБЯЗАН возмутиться! Скажи: 'Ах, катастрофа! Где ваши кроссовки?! В этом же невозможно бегать! Мне пришлось надеть на вас классические туфли, потому что в вашем шкафу нет ничего спортивного!'.\n\n"
        f"ВАЖНОЕ ОГРАНИЧЕНИЕ: Твой комментарий должен быть не длиннее {max_chars} символов!"
    )
    # Если условие катастрофы выполнено — усиливаем подсказку явно
    if is_cold_catastrophe:
        system_prompt += (
            "\n\n⚠️ ВНИМАНИЕ: Сейчас на улице холодно, а сумма тепла вещей меньше 8. "
            "ПУНКТ 5 ОБЯЗАТЕЛЕН — ругай гардероб пользователя и предупреждай о замерзании!"
        )

    user_prompt = (
        f"Место: {location}\n"
        f"Погода: {desc}, {temp:.0f}°C\n"
        f"Повод: {occasion}\n"
        f"Сумма тепла вещей: {total_warmth} (из {len(items) * 5} возможных)\n"
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