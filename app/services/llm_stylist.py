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
    weather: dict,
    location: str,
    occasion: str,
    items: list,
    relaxed_categories: set[str] | list[str] = None,
    max_chars: int = 300,
) -> str | None:
    """Сгенерировать остроумный комментарий-рекомендацию в стиле Александра Васильева.

    relaxed_categories — категории, для которых в гардеробе нет вещей
    подходящего стиля. Если переданы, Васильев ругает эти категории.
    """
    if _llm is None or not items:
        return None

    temp = weather.get("feels_like", weather.get("temp", 15))
    desc = weather.get("description", "")

    items_text = "\n".join(
        f"- {i.category} / {i.type} "
        f"(цвет: {', '.join(get_colors(i)) if get_colors(i) else 'неизвестен'}, "
        f"материал: {i.material or '—'}, "
        f"тепло: {i.warmth_level}/5, стиль: {i.formal_level})"
        for i in items
    )

    # Сумма тепла — чтобы LLM могла вычислить правило пункта 5
    total_warmth = sum(i.warmth_level for i in items)
    is_cold_catastrophe = temp < 10 and total_warmth < 8

    # Инструкция про плохой стиль (флаги relaxed_categories)
    complaint = ""
    if relaxed_categories:
        cats = ", ".join(relaxed_categories)
        complaint = (
            f"6. ВНИМАНИЕ! Для категорий ({cats}) в шкафу нет вещей подходящего стиля. "
            "Тебе пришлось взять то, что есть. Ты ОБЯЗАН возмутиться этим! "
            "Скажи, что это странный набор, это не по правилам этикета, "
            "и человеку будет неудобно.\n"
        )

    system_prompt = (
        "Ты — великий историк моды Александр Васильев. Твоя манера речи: театральная, экспрессивная, "
        "с восклицаниями ('Ах, Боже мой!', 'Это просто катастрофа!', 'Восхитительно!', 'Браво!'), "
        "с легкой иронией и французским шармом.\n\n"
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО слово 'Шарман'.\n\n"
        "Ты только что лично подобрал этот комплект для пользователя. Говори только от первого лица: "
        "'Я подобрал вам...', 'Мне пришлось выбрать...'.\n\n"
        "В твоем комментарии ОБЯЗАТЕЛЬНО должны быть:\n"
        "1. Место и погода.\n"
        "2. Повод.\n"
        "3. Названия и цвета выбранных вещей (упомяни цвет!).\n"
        "4. Упоминание известной достопримечательности этого места.\n"
        "5. ЕСЛИ НА УЛИЦЕ ХОЛОДНО (ниже +10°C), А СУММА ТЕПЛА ВЕЩЕЙ МЕНЬШЕ 8 — "
        "назови это катастрофой и скажи, что человек замерзнет, потому что в шкафу нет теплых вещей.\n"
        f"{complaint}\n"
        f"ВАЖНОЕ ОГРАНИЧЕНИЕ: Твой комментарий должен быть не длиннее {max_chars} символов!"
    )

    # Если условие холодной катастрофы выполнено — усиливаем подсказку явно
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


async def generate_chat_response(user_text: str) -> str | None:
    """Сгенерировать остроумный ответ на любую фразу пользователя."""
    if _llm is None:
        return "Ах, Боже мой! У меня сегодня нет вдохновения..."

    system_prompt = (
        "Ты — великий историк моды Александр Васильев. Твоя манера речи: театральная, экспрессивная, "
        "с восклицаниями ('Ах, Боже мой!', 'Это просто катастрофа!', 'Восхитительно!', 'Браво!'), "
        "с легкой иронией и французским шармом.\n\n"
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО слово 'Шарман'.\n\n"
        "Пользователь задает тебе случайный вопрос или пишет фразу, не связанную с выбором одежды. "
        "Ответь ему остроумно, в своем неповторимом стиле. Можешь похвалить его вкус, "
        "пожаловаться на современные нравы, рассказать короткую байку из мира моды или просто "
        "эффектно пошутить. Ответ должен быть коротким (1-2 предложения)."
    )

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text)
        ])
        return response.content.strip()
    except Exception:
        logger.exception("Ошибка генерации свободного ответа")
        return "Ах, я вас совершенно не понимаю! Наберите /info, чтобы я напомнил вам правила хорошего тона."