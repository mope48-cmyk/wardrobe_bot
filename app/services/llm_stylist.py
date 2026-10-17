"""Выбор комплекта одежды через GigaChat с предварительной фильтрацией."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from app.config import settings
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
        temperature=0.1, # Делаем модель более строгой и предсказуемой
        max_tokens=500,
    )
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — LLM-селектор отключён")


SYSTEM_PROMPT = """Ты — опытный стилист. Тебе дан ФИЛЬТРОВАННЫЙ гардероб пользователя (только лучшие вещи под текущую погоду) и сама погода.

Твоя задача — собрать ОДИН готовый комплект.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. В комплект ОБЯЗАТЕЛЬНО должны входить: 1 верх, 1 низ (или платье), 1 обувь.
2. Обрати ВНИМАНИЕ на температуру. Если на улице +30°C, НЕЛЬЗЯ выбирать тёплые вещи (свитера, ботинки), даже если они есть в списке.
3. Учитывай waterproof при осадках.
4. Учитывай formal_level: работа/встреча → business, прогулка → casual, спорт → sport.

ВАЖНЕЙШЕЕ ПРАВИЛО:
Ты должен использовать ТОЛЬКО те ID вещей, которые перечислены в списке ниже. 
КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать ID, брать их из примеров или использовать те, которых нет в предоставленном списке. 
ID — это уникальные числа, привязанные к конкретным вещам.

ФОРМАТ ОТВЕТА — СТРОГО JSON без markdown, без текста до и после. 
Структура: 
{"item_ids": [СПИСОК_ЧИСЕЛ_ИЗ_СПИСКА], "explanation": "Текст объяснения"}

В поле `explanation` опиши ТОЛЬКО те вещи, ID которых ты указал в `item_ids`. 
Называй вещи ИМЕННО ТАК, как они написаны в списке.
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

# Функция _parse_llm_response остается без изменений
def _parse_llm_response(raw: str) -> dict | None:
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

    return {"item_ids": ids, "explanation": explanation.strip()}