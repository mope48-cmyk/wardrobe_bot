"""Выбор комплекта одежды через GigaChat с валидацией и fallback."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from app.config import settings

logger = logging.getLogger(__name__)

_llm = None

if settings.GIGACHAT_CREDENTIALS:
    _llm = GigaChat(
        credentials=settings.GIGACHAT_CREDENTIALS,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2",
        ca_bundle_file="/app/certs/russian_trusted_root_ca.pem",
        temperature=0.3,
        max_tokens=500,
    )
else:
    logger.warning("GIGACHAT_CREDENTIALS не задан — LLM-селектор отключён")


SYSTEM_PROMPT = """Ты — опытный стилист. Тебе дан гардероб пользователя (у каждой вещи есть ID, категория, тип, цвета, материал, warmth, waterproof, formal, сезон, слой) и текущая погода с поводом.

Твоя задача — собрать ОДИН готовый комплект.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. В комплект ОБЯЗАТЕЛЬНО должны входить:
   - хотя бы одна вещь категории «верх»
   - хотя бы одна вещь категории «низ» (либо одно платье)
   - хотя бы одна вещь категории «обувь»
   Если какой-то из этих категорий НЕТ в гардеробе — не включай её, но и не выдумывай ID.
2. При холоде (<10°C) добавь второй слой (mid) или верхний слой (outer), если они есть в гардеробе.
3. При осадках выбирай верх и обувь с waterproof=true.
4. Повод: работа/встреча → business, прогулка → casual, спорт → sport.

ВАЖНЕЙШЕЕ ПРАВИЛО ДЛЯ ОБЪЯСНЕНИЯ:
В поле `explanation` опиши ТОЛЬКО те вещи, ID которых ты указал в `item_ids`. 
Если ты выбрал водолазку, джинсы и ботинки, напиши именно про них. 
КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО упоминать в тексте вещи, которых нет в списке выбранных (например, не пиши про рубашку или свитер, если ты их не выбрал).

ФОРМАТ ОТВЕТА — СТРОГО JSON без markdown, без текста до и после:
{"item_ids": [49, 51, 58], "explanation": "На улице прохладно, поэтому я выбрал тёплую водолазку. Джинсы и ботинки — отличный выбор для прогулки."}

Используй ТОЛЬКО ID из предоставленного списка."""


async def select_outfit_with_llm(
    weather: dict, occasion: str, items: list
) -> dict | None:
    """Выбрать комплект через GigaChat.

    Возвращает {"item_ids": [int], "explanation": str} или None при ошибке.
    """
    if _llm is None or not items:
        return None

    temp = weather.get("feels_like", weather.get("temp", 15))
    desc = weather.get("description", "")
    wind = weather.get("wind_speed", 0)
    has_precip = weather.get("has_precipitation", False)

    items_lines = []
    for item in items:
        try:
            colors_list = json.loads(item.colors) if item.colors else []
        except Exception:
            colors_list = []
        colors_str = ", ".join(colors_list) if colors_list else "—"

        items_lines.append(
            f"id={item.id} | {item.category}/{item.type} | "
            f"цвета: {colors_str} | материал: {item.material or '—'} | "
            f"тепло: {item.warmth_level}/5 | "
            f"waterproof: {'да' if item.waterproof else 'нет'} | "
            f"стиль: {item.formal_level} | сезон: {item.season} | "
            f"слой: {getattr(item, 'layer', 'base')}"
        )

    items_text = "\n".join(items_lines)
    precip_note = ", осадки" if has_precip else ""

    user_prompt = (
        f"Погода: {desc}, ощущается как {temp:.0f}°C, "
        f"ветер {wind:.0f} м/с{precip_note}.\n"
        f"Повод: {occasion}.\n\n"
        f"Гардероб ({len(items)} вещей):\n{items_text}\n\n"
        "Верни JSON с выбранными ID и объяснением."
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
        logger.warning("LLM вернула не-JSON: %r", raw[:200])
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