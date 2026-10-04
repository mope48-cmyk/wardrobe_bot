"""Узлы графа LangGraph."""

import logging

from langchain_core.messages import AIMessage

from app.db.repository import add_item, get_items
from app.graph.state import BotState
from app.services.outfit import select_outfit
from app.services.vision import classify_clothing
from app.services.weather import geocode_city, get_weather

logger = logging.getLogger(__name__)


# ---------- Клавиатуры ----------

# Спека клавиатуры = список рядов, каждый ряд — список подписей кнопок.
STEP_KEYBOARDS: dict[str, list[list[str]]] = {
    "awaiting_category": [["верх", "низ"], ["обувь", "аксессуар"]],
    "awaiting_warmth": [["1", "2", "3"], ["4", "5"]],
    "awaiting_waterproof": [["да", "нет"]],
    "awaiting_formal": [["casual", "business", "sport"]],
    "awaiting_season": [["лето", "демисезон"], ["зима", "универсальная"]],
    "awaiting_confirm_category": [["да", "нет"]],
}

OCCASION_KEYBOARD = [["работа", "прогулка"], ["спорт", "встреча"], ["другое"]]

REMOVE_KB = "remove"

# Главное меню. Используется после /start, /cancel, сохранения вещи и т.п.
MENU_KEYBOARD = [
    ["/add", "/list"],
    ["/outfit", "/help"],
    ["/cancel"],
]


def _kb(spec) -> dict:
    """Упаковать спеку клавиатуры в additional_kwargs.

    spec — список рядов (кнопки) или строка "remove" (убрать клавиатуру)
    или None (не трогать клавиатуру).
    """
    if spec is None:
        return {}
    return {"reply_keyboard": spec}


def _kb_for_step(step: str) -> dict:
    """Клавиатура для конкретного шага. Если для шага кнопок нет —
    просим убрать клавиатуру, чтобы не оставалась старая."""
    kb = STEP_KEYBOARDS.get(step)
    if kb is None:
        return _kb(REMOVE_KB)
    return _kb(kb)


# ---------- Тексты ----------

WELCOME_TEXT = (
    "Привет! Я помогу каталогизировать твой гардероб "
    "и подбирать одежду по погоде.\n\n"
    "Доступные команды:\n"
    "/add — добавить вещь\n"
    "/list — показать гардероб\n"
    "/outfit — подобрать комплект по погоде\n"
    "/help — справка\n"
    "/cancel — отменить текущее действие"
)

HELP_TEXT = (
    "Что я умею:\n\n"
    "• /add — добавить вещь. Пришлите фото, я попробую распознать категорию и тип.\n\n"
    "• /list — показать все вещи в вашем гардеробе.\n\n"
    "• /outfit — подобрать комплект по погоде.\n\n"
    "• /cancel — отменить текущий диалог."
)


# ---------- Статические команды ----------

async def start_node(state: BotState) -> dict:
    logger.info("start_node: user_id=%s", state.get("user_id"))
    return {
        "messages": [
            AIMessage(content=WELCOME_TEXT, additional_kwargs=_kb(MENU_KEYBOARD))
        ],
        "intent": None, "step": None, "draft_item": None,
    }


async def help_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content=HELP_TEXT)]}


async def cancel_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Хорошо, отменил. Что делаем дальше?",
                additional_kwargs=_kb(MENU_KEYBOARD),
            )
        ],
        "intent": None, "step": None, "draft_item": None,
    }


async def fallback_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content="Не понял. Наберите /help.")]}


async def stub_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content="Эта команда появится позже.")]}


# ---------- /list ----------

async def list_node(state: BotState) -> dict:
    user_id = state["user_id"]
    items = await get_items(user_id)

    if not items:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Ваш гардероб пока пуст.\n\n"
                        "Добавьте первую вещь командой /add."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None, "draft_item": None,
        }

    total = len(items)
    messages: list[AIMessage] = [
        AIMessage(
            content=f"В гардеробе <b>{total}</b> вещей:",
            additional_kwargs=_kb(MENU_KEYBOARD),
        )
    ]

    for item in items:
        waterproof_mark = "💧 " if item.waterproof else ""
        caption = (
            f"<b>{item.category} / {item.type}</b>\n"
            f"цвет: {item.color}\n"
            f"{waterproof_mark}тепло: {item.warmth_level}/5, "
            f"стиль: {item.formal_level}, сезон: {item.season}"
        )
        messages.append(
            AIMessage(
                content=caption,
                additional_kwargs={"photo_file_id": item.photo_file_id},
            )
        )

    return {
        "messages": messages,
        "step": None, "intent": None, "draft_item": None,
    }


# ---------- /add ----------

VALID_CATEGORIES = {"верх", "низ", "обувь", "аксессуар"}
VALID_FORMAL = {"casual", "business", "sport"}
VALID_SEASON = {"лето", "демисезон", "зима", "универсальная"}

STEP_TO_FIELD = {
    "awaiting_category": "category",
    "awaiting_type": "type",
    "awaiting_color": "color",
    "awaiting_warmth": "warmth_level",
    "awaiting_waterproof": "waterproof",
    "awaiting_formal": "formal_level",
    "awaiting_season": "season",
}

NEXT_STEP = {
    "awaiting_category": (
        "awaiting_type",
        "Уточните тип вещи (например: футболка, джинсы, куртка, кроссовки):",
    ),
    "awaiting_type": ("awaiting_color", "Какого цвета вещь?"),
    "awaiting_color": (
        "awaiting_warmth",
        "Насколько вещь тёплая? Оцените от 1 (очень лёгкая) до 5 (очень тёплая):",
    ),
    "awaiting_warmth": (
        "awaiting_waterproof",
        "Эта вещь промокнет под дождём?",
    ),
    "awaiting_waterproof": (
        "awaiting_formal",
        "Для какого случая вещь? Варианты: casual, business, sport:",
    ),
    "awaiting_formal": (
        "awaiting_season",
        "Для какого сезона? Варианты: лето, демисезон, зима, универсальная:",
    ),
}


def _validate_answer(step: str, text: str) -> tuple[bool, object, str]:
    text = text.strip()
    lower = text.lower()

    if step == "awaiting_category":
        if lower in VALID_CATEGORIES:
            return True, lower, ""
        return False, None, "Выберите: верх, низ, обувь или аксессуар."

    if step == "awaiting_type":
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Напишите название типа вещи."

    if step == "awaiting_color":
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Напишите цвет вещи текстом."

    if step == "awaiting_warmth":
        try:
            n = int(text)
            if 1 <= n <= 5:
                return True, n, ""
        except ValueError:
            pass
        return False, None, "Введите число от 1 до 5."

    if step == "awaiting_waterproof":
        # Вопрос: "Эта вещь промокнет под дождём?"
        # "да" (промокнет) → waterproof = False
        # "нет" (не промокнет) → waterproof = True
        if lower in ("да", "yes", "true", "1"):
            return True, False, ""
        if lower in ("нет", "no", "false", "0"):
            return True, True, ""
        return False, None, "Ответьте да или нет."

    if step == "awaiting_formal":
        if lower in VALID_FORMAL:
            return True, lower, ""
        return False, None, "Выберите: casual, business или sport."

    if step == "awaiting_season":
        if lower in VALID_SEASON:
            return True, lower, ""
        return False, None, "Выберите: лето, демисезон, зима или универсальная."

    return False, None, "Неизвестный шаг."


async def add_start_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Пришлите, пожалуйста, фотографию вещи.",
                additional_kwargs=_kb(REMOVE_KB),
            )
        ],
        "intent": "add", "step": "awaiting_photo", "draft_item": {},
    }


async def add_expect_photo_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(content="Жду фотографию вещи. Или /cancel для отмены.")
        ],
    }


async def add_photo_node(state: BotState) -> dict:
    file_id = state.get("input_photo_file_id")
    image_bytes = state.get("input_photo_bytes")

    draft = dict(state.get("draft_item") or {})
    draft["photo_file_id"] = file_id

    if not image_bytes:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Не удалось обработать фото.\n\n"
                        "Что это за вещь? Напишите категорию или выберите:"
                    ),
                    additional_kwargs=_kb_for_step("awaiting_category"),
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    result = await classify_clothing(image_bytes)

    if not result.get("ok") or result.get("category") == "другое":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Не смог распознать вещь на фото.\n\n"
                        "Что это за вещь? Напишите категорию или выберите:"
                    ),
                    additional_kwargs=_kb_for_step("awaiting_category"),
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    draft["predicted_category"] = result["category"]
    draft["predicted_type"] = result["type"]
    draft["predicted_color"] = result["color"]
    draft["predicted_warmth"] = result["warmth_level"]
    draft["predicted_waterproof"] = result["waterproof"]
    draft["predicted_formal"] = result["formal_level"]
    draft["predicted_season"] = result["season"]
    draft["confidence"] = result["confidence"]

    pct = int(result["confidence"] * 100)
    waterproof_str = "да" if result["waterproof"] else "нет"

    return {
        "messages": [
            AIMessage(
                content=(
                    "Я распознал вещь. Проверьте, всё ли верно:\n\n"
                    f"• Категория: <b>{result['category']}</b>\n"
                    f"• Тип: <b>{result['type']}</b>\n"
                    f"• Цвет: <b>{result['color']}</b>\n"
                    f"• Тепло: <b>{result['warmth_level']}/5</b>\n"
                    f"• Водонепроницаемая: <b>{waterproof_str}</b>\n"
                    f"• Стиль: <b>{result['formal_level']}</b>\n"
                    f"• Сезон: <b>{result['season']}</b>\n\n"
                    f"<i>Уверенность распознавания: {pct}%</i>\n\n"
                    "Если всё верно — ответьте <b>да</b>, и я сразу сохраню вещь.\n"
                    "Если нужно исправить — ответьте <b>нет</b> и введите всё вручную."
                ),
                additional_kwargs=_kb_for_step("awaiting_confirm_category"),
            )
        ],
        "step": "awaiting_confirm_category",
        "draft_item": draft,
    }


async def add_confirm_category_node(state: BotState) -> dict:
    text = (state.get("input_text") or "").strip().lower()
    draft = dict(state.get("draft_item") or {})

    if text in ("да", "yes", "верно", "ага", "ok", "+"):
        draft["category"] = draft["predicted_category"]
        draft["type"] = draft["predicted_type"]
        draft["color"] = draft.get("predicted_color") or ""
        draft["warmth_level"] = draft["predicted_warmth"]
        draft["waterproof"] = draft["predicted_waterproof"]
        draft["formal_level"] = draft["predicted_formal"]
        draft["season"] = draft["predicted_season"]

        try:
            await add_item(
                user_id=state["user_id"],
                photo_file_id=draft["photo_file_id"],
                category=draft["category"],
                type=draft["type"],
                color=draft["color"],
                material="",
                warmth_level=draft["warmth_level"],
                waterproof=draft["waterproof"],
                formal_level=draft["formal_level"],
                season=draft["season"],
            )
        except Exception as e:
            logger.exception("Ошибка сохранения вещи")
            return {
                "messages": [
                    AIMessage(
                        content=f"Не удалось сохранить: {e}",
                        additional_kwargs=_kb(MENU_KEYBOARD),
                    )
                ],
                "step": None, "draft_item": None, "intent": None,
            }

        waterproof_str = "да" if draft["waterproof"] else "нет"
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Готово! Сохранил вещь:\n"
                        f"• {draft['category']} / {draft['type']}\n"
                        f"• цвет: {draft['color']}\n"
                        f"• тепло: {draft['warmth_level']}/5\n"
                        f"• водонепроницаемая: {waterproof_str}\n"
                        f"• стиль: {draft['formal_level']}\n"
                        f"• сезон: {draft['season']}\n\n"
                        "Добавьте ещё вещь командой /add."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "draft_item": None, "intent": None,
        }

    if text in ("нет", "no", "неверно", "-"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Хорошо, вводим вручную.\n\n"
                        "Категория: <b>верх</b>, <b>низ</b>, <b>обувь</b> "
                        "или <b>аксессуар</b>?"
                    ),
                    additional_kwargs=_kb_for_step("awaiting_category"),
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    return {
        "messages": [
            AIMessage(
                content="Ответьте <b>да</b> (сохранить) или <b>нет</b> (ввести заново).",
                additional_kwargs=_kb_for_step("awaiting_confirm_category"),
            )
        ],
    }


async def add_attribute_node(state: BotState) -> dict:
    step = state.get("step")
    text = state.get("input_text") or ""

    logger.info("add_attribute_node: step=%s text=%r", step, text)

    valid, value, error = _validate_answer(step, text)
    if not valid:
        return {"messages": [AIMessage(content=error)]}

    draft = dict(state.get("draft_item") or {})
    draft[STEP_TO_FIELD[step]] = value

    if step == "awaiting_season":
        try:
            await add_item(
                user_id=state["user_id"],
                photo_file_id=draft["photo_file_id"],
                category=draft["category"],
                type=draft["type"],
                color=draft["color"],
                material="",
                warmth_level=draft["warmth_level"],
                waterproof=draft["waterproof"],
                formal_level=draft["formal_level"],
                season=draft["season"],
            )
        except Exception as e:
            logger.exception("Ошибка сохранения вещи")
            return {
                "messages": [
                    AIMessage(
                        content=f"Не удалось сохранить: {e}",
                        additional_kwargs=_kb(MENU_KEYBOARD),
                    )
                ],
                "step": None, "draft_item": None, "intent": None,
            }

        waterproof_str = "да" if draft["waterproof"] else "нет"
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Готово! Сохранил вещь:\n"
                        f"• {draft['category']} / {draft['type']}\n"
                        f"• цвет: {draft['color']}\n"
                        f"• тепло: {draft['warmth_level']}/5\n"
                        f"• водонепроницаемая: {waterproof_str}\n"
                        f"• стиль: {draft['formal_level']}\n"
                        f"• сезон: {draft['season']}\n\n"
                        "Добавьте ещё вещь командой /add."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "draft_item": None, "intent": None,
        }

    next_step, question = NEXT_STEP[step]
    return {
        "messages": [
            AIMessage(
                content=question,
                additional_kwargs=_kb_for_step(next_step),
            )
        ],
        "step": next_step,
        "draft_item": draft,
    }


# ---------- /outfit ----------

VALID_OCCASIONS = {"работа", "прогулка", "спорт", "встреча", "другое"}


async def outfit_start_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "Чтобы подобрать комплект, мне нужно знать, где вы находитесь.\n\n"
                    "Отправьте <b>геолокацию</b> (скрепка → Геопозиция) "
                    "или напишите название города."
                ),
                additional_kwargs=_kb(REMOVE_KB),
            )
        ],
        "intent": "outfit",
        "step": "awaiting_location",
        "draft_item": None,
        "location": None,
        "occasion": None,
        "weather": None,
        "outfit": None,
    }


async def outfit_location_node(state: BotState) -> dict:
    loc = state.get("input_location")
    text = (state.get("input_text") or "").strip()

    if loc:
        location = {"lat": loc["lat"], "lon": loc["lon"], "city": None}
    elif text:
        geo = await geocode_city(text)
        if not geo:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Не смог найти этот город. Попробуйте ещё раз "
                            "или отправьте геолокацию."
                        )
                    )
                ],
            }
        location = geo
    else:
        return {
            "messages": [
                AIMessage(
                    content="Жду геолокацию или название города. Или /cancel."
                )
            ],
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    "Понял.\n\n"
                    "Какой повод? Выберите один из вариантов."
                ),
                additional_kwargs=_kb(OCCASION_KEYBOARD),
            )
        ],
        "step": "awaiting_occasion",
        "location": location,
    }


async def outfit_occasion_node(state: BotState) -> dict:
    text = (state.get("input_text") or "").strip().lower()

    if text not in VALID_OCCASIONS:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Выберите повод из предложенных вариантов "
                        "(или /cancel для отмены)."
                    ),
                    additional_kwargs=_kb(OCCASION_KEYBOARD),
                )
            ],
        }

    loc = state.get("location") or {}
    if not loc:
        return {
            "messages": [
                AIMessage(
                    content="Что-то пошло не так с местоположением. Начните заново: /outfit.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }

    weather = await get_weather(loc["lat"], loc["lon"])
    if not weather:
        return {
            "messages": [
                AIMessage(
                    content="Не удалось получить погоду. Попробуйте позже или начните заново: /outfit.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None, "location": None,
        }

    items = await get_items(state["user_id"])
    if not items:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Ваш гардероб пуст. Добавьте вещи через /add, "
                        "затем повторите /outfit."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None, "location": None,
            "weather": weather,
        }

    outfit = select_outfit(list(items), weather, text)

    city_str = f" ({loc['city']})" if loc.get("city") else ""
    temp = weather["temp"]
    desc = weather["description"]
    header = f"Погода{city_str}: {desc}, {temp:.0f}°C.\nПовод: {text}."

    if not outfit:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"{header}\n\n"
                        "К сожалению, в вашем гардеробе нет подходящих вещей. "
                        "Добавьте их через /add."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None, "location": None,
            "weather": weather,
        }

    messages: list[AIMessage] = [
        AIMessage(content=f"{header}\n\nВот что предлагаю надеть:")
    ]

    for category, item in outfit.items():
        waterproof_mark = "💧 " if item.waterproof else ""
        caption = (
            f"<b>{item.category} / {item.type}</b>\n"
            f"цвет: {item.color}\n"
            f"{waterproof_mark}тепло: {item.warmth_level}/5, "
            f"стиль: {item.formal_level}, сезон: {item.season}"
        )
        messages.append(
            AIMessage(
                content=caption,
                additional_kwargs={"photo_file_id": item.photo_file_id},
            )
        )

    # Финальное сообщение с главным меню
    messages.append(
        AIMessage(
            content="Если хотите — подберите ещё раз: /outfit.",
            additional_kwargs=_kb(MENU_KEYBOARD),
        )
    )

    return {
        "messages": messages,
        "step": None, "intent": None,
        "location": None, "weather": weather,
        "occasion": text,
    }