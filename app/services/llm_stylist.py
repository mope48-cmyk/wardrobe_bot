"""Выбор комплекта одежды через GigaChat с предварительной фильтрацией."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from app.config import settings
from app.db.repository import get_colors
# Импортируем функцию оценки из вашего outfit.py
from app.services.outfit import _score, OCCASION_TO_FORMAL

logger = logging.getLogger(__name__)

_llm = None

if settings.GIGACHAT_CREDENTIALS:
    _llm = GigaChat(
        credentials=settings.GIGACHAT_CREDENTIALS,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2-Pro",  # <--- Меняем на Pro!
        ca_bundle_file="/app/certs/russian_trusted_root_ca.pem",
        temperature=0.1,  # Делаем модель более строгой и предсказуемой
        max_tokens=500,
    )
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — LLM-селектор отключён")


SYSTEM_PROMPT = """Ты — опытный стилист. Тебе дан ФИЛЬТРОВАННЫЙ гардероб пользователя (только лучшие вещи под текущую погоду) и сама погода.

Твоя задача — собрать ОДИН готовый комплект.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. В комплект ОБЯЗАТЕЛЬНО должны входить: 1 верх, 1 низ (или платье), 1 обувь.
2. Слои по температуре:
   - Если на улице < 0°C, нужно 3 слоя одежды (базовый, средний, верхний). Выбери соответствующие вещи с нужным слоем.
   - Если от 0°C до 15°C, нужно 2 слоя (базовый + средний или базовый + верхний).
   - Если > 15°C, достаточно 1 слоя (базовый).
3. Цветовой бюджет. В образе не должно быть больше 3 баллов за цвет.
   - Серый (любой оттенок), чёрный и белый — 0.5 балла.
   - Любой другой цвет — 1 балл.
   Суммируй баллы всех выбранных вещей. Не превышай лимит в 3 балла!
4. Учитывай waterproof при осадках.
5. Учитывай formal_level: работа/встреча → business, прогулка → casual, спорт → sport.

ВАЖНЕЙШЕЕ ПРАВИЛО:
Ты должен использовать ТОЛЬКО те ID вещей, которые перечислены в списке ниже. 
КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать ID, брать их из примеров или использовать те, которых нет в предоставленном списке. 

ФОРМАТ ОТВЕТА — СТРОГО JSON без markdown:
{"item_ids": [СПИСОК_ЧИСЕЛ_ИЗ_СПИСКА], "explanation": "Текст объяснения"}

В поле `explanation` опиши ТОЛЬКО те вещи, ID которых ты указал в `item_ids`. Называй вещи ИМЕННО ТАК, как они написаны в списке.
"""


async def select_outfit_with_llm(
    weather: dict, occasion: str, items: list
) -> dict | None:
    if _llm is None or not items:
        return None

    formal = OCCASION_TO_FORMAL.get(occasion)

    # 1. ФИЛЬТРАЦИЯ: группируем и оставляем только топ-3 по рейтингу _score
    by_cat = {"верх": [], "низ": [], "обувь": [], "аксессуар": []}
    for item in items:
        if item.category in by_cat:
            by_cat[item.category].append(item)

    filtered_items = []
    for cat, cat_items in by_cat.items():
        # Сортируем по убыванию рейтинга
        sorted_cat = sorted(
            cat_items,
            key=lambda i: -_score(i, formal, weather, occasion)
        )
        # Берем только 3 лучших в каждой категории
        filtered_items.extend(sorted_cat[:3])

    if not filtered_items:
        return None

    logger.info("После фильтрации осталось %d вещей для LLM", len(filtered_items))

    # 2. Формируем текстовый список для LLM
    items_lines = []
    for item in filtered_items:
        try:
            colors_list = json.loads(item.colors) if item.colors else []
        except Exception:
            colors_list = []
        colors_str = ", ".join(colors_list) if colors_list else "—"

        line = (
            f"id={item.id} | {item.category}/{item.type} | "
            f"цвета: {colors_str} | материал: {item.material or '—'} | "
            f"тепло: {item.warmth_level}/5 | "
            f"waterproof: {'да' if item.waterproof else 'нет'} | "
            f"стиль: {item.formal_level} | сезон: {item.season} | "
            f"слой: {getattr(item, 'layer', 'base')}"
        )
        items_lines.append(line)

    items_text = "\n".join(items_lines)

    temp = weather.get("feels_like", weather.get("temp", 15))
    desc = weather.get("description", "")
    wind = weather.get("wind_speed", 0)
    precip_note = ", осадки" if weather.get("has_precipitation", False) else ""

    user_prompt = (
        f"Погода: {desc}, ощущается как {temp:.0f}°C, ветер {wind:.0f} м/с{precip_note}.\n"
        f"Повод: {occasion}.\n\n"
        f"Доступные вещи (отфильтрованы по погоде):\n{items_text}\n\n"
        "Собери комплект и верни JSON."
    )

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw = response.content.strip()
    except Exception:
        logger.exception("Ошибка GigaChat при выборе комплекта")
        return None

    return _parse_llm_response(raw)


def _parse_llm_response(raw: str) -> dict | None:
    """Извлечь JSON из ответа LLM (может быть в markdown-обёртке)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    if not isinstance(data, dict):
        return None

    ids = data.get("item_ids")
    explanation = data.get("explanation", "")

    if not isinstance(ids, list):
        return None

    try:
        ids = [int(x) for x in ids]
    except (ValueError, TypeError):
        return None

    if not isinstance(explanation, str):
        explanation = ""

    return {"item_ids": ids, "explanation": explanation.strip()}


async def generate_vasiliev_comment(
    weather: dict, location: str, occasion: str, items: list
) -> str | None:
    """Сгенерировать остроумный комментарий в стиле Александра Васильева."""
    if _llm is None or not items:
        return None

    temp = weather.get("feels_like", weather.get("temp", 15))
    desc = weather.get("description", "")

    # Собираем список вещей
    items_text = "\n".join(
        f"- {i.category} / {i.type} "
        f"(цвета: {', '.join(get_colors(i)) if get_colors(i) else '—'}, "
        f"материал: {i.material or '—'})"
        for i in items
    )

    system_prompt = (
        "Ты — великий историк моды Александр Васильев. Твоя манера речи: "
        "театральная, экспрессивная, с восклицаниями ('Ах, Боже мой!', "
        "'Это просто катастрофа!', 'Восхитительно!'), "
        "легкой иронией и французским шармом. Ты обожаешь обсуждать образы, "
        "погоду и поводы.\n\n"
        "Твоя задача — дать короткий (3-4 предложения) остроумный комментарий "
        "к образу пользователя. "
        "Обязательно упомяни место, погоду, повод и сами вещи (называй их так же, "
        "как в списке). "
        "Пиши от первого лица, эмоционально и с юмором."
    )

    user_prompt = (
        f"Место: {location}\n"
        f"Погода: {desc}, {temp:.0f}°C\n"
        f"Повод: {occasion}\n"
        f"Выбранный комплект:\n{items_text}\n\n"
        "Напиши свой вердикт этому образу!"
    )

    try:
        response = await _llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        # Возвращаем только текст, без JSON
        return response.content.strip()
    except Exception:
        logger.exception("Ошибка генерации комментария Васильева")
        return None