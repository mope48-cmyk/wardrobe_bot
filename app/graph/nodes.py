"""Узлы графа LangGraph."""

import logging

from langchain_core.messages import AIMessage

from app.db.repository import (
    add_item,
    delete_item,
    get_item,
    get_items,
    update_item_field,
)
from app.graph.state import BotState
from app.services.outfit import select_outfit
from app.services.vision import classify_clothing
from app.services.weather import geocode_city, get_weather

logger = logging.getLogger(__name__)


# ---------- Клавиатуры ----------

# Спека клавиатуры = список рядов, каждый ряд — список подписей кнопок.
# Шага awaiting_confirm_category здесь нет: его клавиатура динамическая,
# зависит от вариантов распознавания.
STEP_KEYBOARDS: dict[str, list[list[str]]] = {
    "awaiting_category": [["верх", "низ"], ["обувь", "аксессуар"]],
    "awaiting_warmth": [["1", "2", "3"], ["4", "5"]],
    "awaiting_waterproof": [["да", "нет"]],
    "awaiting_formal": [["casual", "business", "sport"]],
    "awaiting_season": [["лето", "демисезон"], ["зима", "универсальная"]],
}

OCCASION_KEYBOARD = [["работа", "прогулка"], ["спорт", "встреча"], ["другое"]]

REMOVE_KB = "remove"

# Главное меню. Используется после /start, /cancel, сохранения вещи и т.п.
# Подписи на русском с эмодзи. Маршрутизация по ним — в builder.route_condition.
MENU_KEYBOARD = [
    ["➕ Добавить вещь"],
    ["👕 Мой гардероб", "🌤 Подобрать образ"],
    ["❓ Помощь", "❌ Отмена"],
]


def _kb(spec) -> dict:
    """Упаковать спеку клавиатуры в additional_kwargs.

    spec — список рядов (кнопки), или строка "remove" (убрать клавиатуру),
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


def _confirm_keyboard(options: list[dict]) -> list[list[str]]:
    """Собрать клавиатуру с кнопками типов из options.

    Максимум 3 типа + кнопка "другое". Ряды по 2 кнопки,
    чтобы длинные названия («классическая рубашка») помещались.
    """
    buttons = [opt["type"] for opt in options[:3]]
    buttons.append("другое")

    rows: list[list[str]] = []
    for i in range(0, len(buttons), 2):
        rows.append(buttons[i:i + 2])
    return rows


async def _persist_item(state: BotState, draft: dict) -> dict:
    """Сохранить вещь в БД и вернуть готовый ответ для пользователя.

    Используется и после подтверждения распознавания,
    и после ручного ввода.
    """
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


# ---------- /list: постраничный просмотр ----------

LIST_PREVIEW_LEN = 60  # сколько символов типа показывать в заголовке


def _item_caption(item) -> str:
    """Подпись под фото вещи."""
    waterproof_mark = "💧 " if item.waterproof else ""
    return (
        f"<b>{item.category} / {item.type}</b>\n"
        f"цвет: {item.color}\n"
        f"{waterproof_mark}тепло: {item.warmth_level}/5, "
        f"стиль: {item.formal_level}, сезон: {item.season}"
    )


def _list_keyboard(index: int, total: int) -> list[list[dict]]:
    """Inline-клавиатура для просмотра одной вещи.

    На границах prev/next становятся noop-кнопками — они не меняют
    состояние, но показывают подсказку через callback.answer (в handlers).
    """
    prev_cb = "list:prev" if index > 0 else "list:noop:first"
    next_cb = "list:next" if index < total - 1 else "list:noop:last"

    return [
        [
            {"text": "✏️ Редактировать", "callback_data": "list:edit"},
            {"text": "🗑 Удалить", "callback_data": "list:delete"},
        ],
        [
            {"text": "⬅️ Назад", "callback_data": prev_cb},
            {"text": "Вперёд ➡️", "callback_data": next_cb},
        ],
    ]


async def _render_current(state: BotState, edit_mode: str = "media") -> dict:
    """Отрисовать текущую вещь из list_ids[list_index].

    Универсальная функция: используется и для /list, и для поиска,
    и для любого другого списка, положенного в list_ids.

    edit_mode:
      - "media"   → handlers вызовет edit_message_media
                    (подходит, когда меняется фото)
      - "caption" → handlers вызовет edit_message_caption
                    (когда фото то же, меняется только текст и кнопки)
    """
    ids = list(state.get("list_ids") or [])
    index = state.get("list_index") or 0
    edit_message_id = state.get("edit_message_id")

    if not ids:
        return {
            "messages": [
                AIMessage(
                    content="Список пуст.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
            "list_ids": None, "list_index": 0,
            "edit_message_id": None,
        }

    index = max(0, min(index, len(ids) - 1))
    total = len(ids)
    item_id = ids[index]
    item = await get_item(item_id)

    if item is None:
        # Вещь удалили — убираем её из списка
        ids.pop(index)
        if not ids:
            return {
                "messages": [
                    AIMessage(
                        content="В гардеробе больше нет вещей.",
                        additional_kwargs=_kb(MENU_KEYBOARD),
                    )
                ],
                "step": None, "intent": None,
                "list_ids": None, "list_index": 0,
                "edit_message_id": None,
            }
        new_index = min(index, len(ids) - 1)
        return await _render_current({
            **state,
            "list_ids": ids,
            "list_index": new_index,
        })

    header = f"📋 Вещь <b>{index + 1}</b> из <b>{total}</b>\n\n"
    caption = header + _item_caption(item)

    additional: dict = {
        "photo_file_id": item.photo_file_id,
        "inline_keyboard": _list_keyboard(index, total),
        "edit_mode": edit_mode,
    }
    if edit_message_id is not None:
        additional["edit_message_id"] = edit_message_id

    return {
        "messages": [AIMessage(content=caption, additional_kwargs=additional)],
        "step": "list_view",
        "intent": "list",
        "list_ids": ids,
        "list_index": index,
        "list_source": state.get("list_source") or "list",
        "edit_message_id": edit_message_id,
    }


async def list_node(state: BotState) -> dict:
    """/list — загружаем все вещи пользователя и показываем первую."""
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
            "step": None, "intent": None,
            "list_ids": None, "list_index": 0,
            "edit_message_id": None,
        }

    ids = [item.id for item in items]
    return await _render_current({
        **state,
        "list_ids": ids,
        "list_index": 0,
        "list_source": "list",
        "edit_message_id": None,  # первое сообщение — новое
    })


async def list_prev_node(state: BotState) -> dict:
    """Перейти к предыдущей вещи."""
    index = state.get("list_index") or 0
    return await _render_current({**state, "list_index": max(0, index - 1)})


async def list_next_node(state: BotState) -> dict:
    """Перейти к следующей вещи."""
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    return await _render_current({
        **state,
        "list_index": min(len(ids) - 1, index + 1),
    })


async def list_noop_node(state: BotState) -> dict:
    """Ничего не делать (границы списка).

    Handlers сам покажет alert через callback.answer — здесь просто
    возвращаем пустой ответ, чтобы граф не падал.
    """
    return {}


# ---------- Редактирование вещи ----------

# Логическое имя поля → колонка в БД
EDIT_FIELD_TO_DB = {
    "category": "category",
    "type": "type",
    "color": "color",
    "warmth": "warmth_level",
    "waterproof": "waterproof",
    "formal": "formal_level",
    "season": "season",
}

# Человекочитаемое имя поля
EDIT_FIELD_LABEL = {
    "category": "Категория",
    "type": "Тип",
    "color": "Цвет",
    "warmth": "Тепло (1–5)",
    "waterproof": "Водонепроницаемость (да/нет)",
    "formal": "Стиль (casual / business / sport)",
    "season": "Сезон (лето / демисезон / зима / универсальная)",
}


def _validate_edit_field(
    field: str, text: str
) -> tuple[bool, object, str]:
    """Проверить ввод пользователя при редактировании поля."""
    text = text.strip()
    lower = text.lower()

    if field == "category":
        if lower in VALID_CATEGORIES:
            return True, lower, ""
        return False, None, "Выберите: верх, низ, обувь или аксессуар."

    if field == "type":
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Слишком коротко. Напишите название типа."

    if field == "color":
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Напишите цвет текстом."

    if field == "warmth":
        try:
            n = int(text)
            if 1 <= n <= 5:
                return True, n, ""
        except ValueError:
            pass
        return False, None, "Введите число от 1 до 5."

    if field == "waterproof":
        if lower in ("да", "yes", "true", "1"):
            return True, True, ""
        if lower in ("нет", "no", "false", "0"):
            return True, False, ""
        return False, None, "Ответьте да или нет."

    if field == "formal":
        if lower in VALID_FORMAL:
            return True, lower, ""
        return False, None, "Выберите: casual, business или sport."

    if field == "season":
        if lower in VALID_SEASON:
            return True, lower, ""
        return False, None, "Выберите: лето, демисезон, зима или универсальная."

    return False, None, "Неизвестное поле."


def _field_value_str(item, field: str) -> str:
    """Текущее значение поля в виде строки для отображения."""
    db_field = EDIT_FIELD_TO_DB[field]
    value = getattr(item, db_field)
    if isinstance(value, bool):
        return "да" if value else "нет"
    return str(value)


async def list_edit_node(state: BotState) -> dict:
    """Показать меню редактирования вещи."""
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    edit_message_id = state.get("edit_message_id")

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего редактировать.")],
            "step": None, "intent": None,
        }

    item = await get_item(ids[index])
    if item is None:
        return {
            "messages": [AIMessage(content="Вещь не найдена.")],
            "step": None, "intent": None,
        }

    caption = (
        "✏️ <b>Редактирование вещи</b>\n\n"
        f"{item.category} / {item.type}\n"
        f"цвет: {item.color}\n"
        f"тепло: {item.warmth_level}/5\n"
        f"водонепроницаемая: {'да' if item.waterproof else 'нет'}\n"
        f"стиль: {item.formal_level}\n"
        f"сезон: {item.season}\n\n"
        "Что изменить?"
    )

    keyboard = [
        [
            {"text": "Категория", "callback_data": "list:edit:category"},
            {"text": "Тип", "callback_data": "list:edit:type"},
        ],
        [
            {"text": "Цвет", "callback_data": "list:edit:color"},
            {"text": "Тепло", "callback_data": "list:edit:warmth"},
        ],
        [
            {"text": "Водонепроницаемость", "callback_data": "list:edit:waterproof"},
        ],
        [
            {"text": "Стиль", "callback_data": "list:edit:formal"},
            {"text": "Сезон", "callback_data": "list:edit:season"},
        ],
        [
            {"text": "❌ Отмена", "callback_data": "list:edit:cancel"},
        ],
    ]

    additional: dict = {
        "photo_file_id": item.photo_file_id,
        "inline_keyboard": keyboard,
        "edit_mode": "caption",
    }
    if edit_message_id is not None:
        additional["edit_message_id"] = edit_message_id

    return {
        "messages": [AIMessage(content=caption, additional_kwargs=additional)],
        "step": "list_edit_menu",
        "intent": "list",
        "list_ids": ids,
        "list_index": index,
        "edit_message_id": edit_message_id,
        "draft_item": None,
    }


async def list_edit_field_node(state: BotState) -> dict:
    """Пользователь выбрал поле — просим ввести новое значение."""
    text = (state.get("input_text") or "").strip()
    # text выглядит как "list:edit:category"
    parts = text.split(":", 2)
    field = parts[2] if len(parts) == 3 else ""

    if field not in EDIT_FIELD_TO_DB:
        return {
            "messages": [AIMessage(content="Неизвестное поле.")],
            "step": None, "intent": None,
        }

    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    edit_message_id = state.get("edit_message_id")

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего редактировать.")],
            "step": None, "intent": None,
        }

    item = await get_item(ids[index])
    if item is None:
        return {
            "messages": [AIMessage(content="Вещь не найдена.")],
            "step": None, "intent": None,
        }

    current_str = _field_value_str(item, field)
    label = EDIT_FIELD_LABEL[field]

    caption = (
        f"✏️ <b>{label}</b>\n\n"
        f"Текущее значение: <b>{current_str}</b>\n\n"
        "Введите новое значение текстом."
    )

    additional: dict = {
        "photo_file_id": item.photo_file_id,
        "inline_keyboard": [
            [{"text": "❌ Отмена", "callback_data": "list:edit:cancel"}],
        ],
        "edit_mode": "caption",
    }
    if edit_message_id is not None:
        additional["edit_message_id"] = edit_message_id

    return {
        "messages": [AIMessage(content=caption, additional_kwargs=additional)],
        "step": "list_edit_input",
        "intent": "list",
        "list_ids": ids,
        "list_index": index,
        "edit_message_id": edit_message_id,
        "draft_item": {"edit_field": field},
    }


async def list_edit_save_node(state: BotState) -> dict:
    """Сохранить новое значение поля и вернуться к просмотру вещи."""
    text = (state.get("input_text") or "").strip()
    draft = state.get("draft_item") or {}
    field = draft.get("edit_field")
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    edit_message_id = state.get("edit_message_id")
    user_id = state["user_id"]

    if not field or field not in EDIT_FIELD_TO_DB:
        return {
            "messages": [
                AIMessage(
                    content="Что-то пошло не так. Начните заново: /list.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
            "list_ids": None, "list_index": 0,
            "edit_message_id": None, "draft_item": None,
        }

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего редактировать.")],
            "step": None, "intent": None,
            "draft_item": None,
        }

    valid, value, error = _validate_edit_field(field, text)
    if not valid:
        # Показываем ошибку поверх той же подсказки, шаг не меняем
        item = await get_item(ids[index])
        current_str = _field_value_str(item, field) if item else "?"
        label = EDIT_FIELD_LABEL[field]

        caption = (
            f"⚠️ {error}\n\n"
            f"<b>{label}</b>\n"
            f"Текущее значение: <b>{current_str}</b>\n\n"
            "Введите новое значение."
        )

        additional: dict = {
            "inline_keyboard": [
                [{"text": "❌ Отмена", "callback_data": "list:edit:cancel"}],
            ],
            "edit_mode": "caption",
        }
        if edit_message_id is not None:
            additional["edit_message_id"] = edit_message_id

        return {
            "messages": [AIMessage(content=caption, additional_kwargs=additional)],
        }

    item_id = ids[index]
    ok = await update_item_field(
        item_id, user_id, EDIT_FIELD_TO_DB[field], value
    )
    if not ok:
        return {
            "messages": [
                AIMessage(
                    content="Не удалось сохранить изменение.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
            "list_ids": None, "list_index": 0,
            "edit_message_id": None, "draft_item": None,
        }

    # Возвращаемся к просмотру той же вещи с обновлёнными данными
    return await _render_current(
        {**state, "draft_item": None},
        edit_mode="caption",
    )


async def list_edit_cancel_node(state: BotState) -> dict:
    """Отменить редактирование — вернуться к просмотру вещи."""
    return await _render_current(
        {**state, "draft_item": None},
        edit_mode="caption",
    )


async def list_delete_node(state: BotState) -> dict:
    """Показать подтверждение удаления текущей вещи.

    Редактирует то же сообщение (edit_mode="caption"), меняя caption
    на вопрос подтверждения и клавиатуру на [✅ Да] [❌ Отмена].
    """
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    edit_message_id = state.get("edit_message_id")

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего удалять.")],
            "step": None, "intent": None,
        }

    item = await get_item(ids[index])
    if item is None:
        return {
            "messages": [AIMessage(content="Вещь не найдена.")],
            "step": None, "intent": None,
        }

    caption = (
        "🗑 <b>Удалить вещь?</b>\n\n"
        f"{item.category} / {item.type}\n"
        f"цвет: {item.color}\n\n"
        "Это действие необратимо."
    )

    additional: dict = {
        "photo_file_id": item.photo_file_id,
        "inline_keyboard": [
            [
                {"text": "✅ Да, удалить", "callback_data": "list:delete_confirm"},
                {"text": "❌ Отмена", "callback_data": "list:delete_cancel"},
            ]
        ],
        "edit_mode": "caption",
    }
    if edit_message_id is not None:
        additional["edit_message_id"] = edit_message_id

    return {
        "messages": [AIMessage(content=caption, additional_kwargs=additional)],
        "step": "list_view",
        "intent": "list",
        "list_ids": ids,
        "list_index": index,
        "edit_message_id": edit_message_id,
    }


async def list_delete_do_node(state: BotState) -> dict:
    """Выполнить удаление и показать следующую вещь."""
    ids = list(state.get("list_ids") or [])
    index = state.get("list_index") or 0
    edit_message_id = state.get("edit_message_id")
    user_id = state["user_id"]

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего удалять.")],
            "step": None, "intent": None,
        }

    item_id = ids[index]
    item = await get_item(item_id)

    deleted = await delete_item(item_id, user_id)
    if not deleted:
        return {
            "messages": [
                AIMessage(
                    content="Не удалось удалить вещь — возможно, её уже нет.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
            "list_ids": None, "list_index": 0,
            "edit_message_id": None,
        }

    # Убираем id из списка
    new_ids = [i for i in ids if i != item_id]

    if not new_ids:
        # Гардероб опустел — новое сообщение (нельзя отредактировать медиа в пустое)
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"🗑 Вещь «{item.category} / {item.type}» удалена.\n\n"
                        "Больше в гардеробе ничего нет."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
            "list_ids": None, "list_index": 0,
            "edit_message_id": None,
        }

    # Определяем индекс следующей вещи:
    # если удалили последнюю — сдвигаемся на предыдущую,
    # иначе остаёмся на том же индексе (там теперь следующая).
    new_index = min(index, len(new_ids) - 1)

    return await _render_current({
        **state,
        "list_ids": new_ids,
        "list_index": new_index,
        "edit_message_id": edit_message_id,
    })


async def list_delete_cancel_node(state: BotState) -> dict:
    """Отменить удаление — вернуться к просмотру текущей вещи.

    edit_mode="caption": фото то же, меняется только caption и кнопки.
    """
    return await _render_current(state, edit_mode="caption")


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

    if not result.get("ok") or not result.get("options"):
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

    options = result["options"]
    draft["options"] = options
    draft["predicted_color"] = result["color"]
    draft["confidence"] = result["confidence"]

    # Формируем список вариантов в тексте
    lines = ["Я распознал вещь. Вот возможные типы:\n"]
    for i, opt in enumerate(options, start=1):
        pct = int(opt["confidence"] * 100)
        if i == 1:
            lines.append(f"{i}. <b>{opt['type']}</b> — {pct}%")
        else:
            lines.append(f"{i}. {opt['type']} — {pct}%")

    lines.append("")
    lines.append(f"Цвет: <b>{result['color']}</b>.")
    lines.append("")
    lines.append(
        "Выберите тип кнопкой ниже. Если ничего не подходит — нажмите "
        "<b>другое</b> и введите вручную."
    )

    return {
        "messages": [
            AIMessage(
                content="\n".join(lines),
                additional_kwargs=_kb(_confirm_keyboard(options)),
            )
        ],
        "step": "awaiting_confirm_category",
        "draft_item": draft,
    }


async def add_confirm_category_node(state: BotState) -> dict:
    text = (state.get("input_text") or "").strip().lower()
    draft = dict(state.get("draft_item") or {})
    options = draft.get("options") or []

    # 1. Пользователь выбрал один из предложенных типов
    for opt in options:
        if opt["type"].lower() == text:
            draft["category"] = opt["category"]
            draft["type"] = opt["type"]
            draft["color"] = draft.get("predicted_color") or ""
            draft["warmth_level"] = opt["warmth_level"]
            draft["waterproof"] = opt["waterproof"]
            draft["formal_level"] = opt["formal_level"]
            draft["season"] = opt["season"]
            return await _persist_item(state, draft)

    # 2. "да" → принять топ-1
    if text in ("да", "yes", "верно", "ага", "ok", "+"):
        if options:
            opt = options[0]
            draft["category"] = opt["category"]
            draft["type"] = opt["type"]
            draft["color"] = draft.get("predicted_color") or ""
            draft["warmth_level"] = opt["warmth_level"]
            draft["waterproof"] = opt["waterproof"]
            draft["formal_level"] = opt["formal_level"]
            draft["season"] = opt["season"]
            return await _persist_item(state, draft)

    # 3. "другое" или "нет" → ручной ввод
    if text in ("другое", "нет", "no", "неверно", "-"):
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

    # 4. Ничего не подошло — повторить
    return {
        "messages": [
            AIMessage(
                content=(
                    "Выберите тип из кнопок ниже или нажмите <b>другое</b>."
                ),
                additional_kwargs=_kb(_confirm_keyboard(options)),
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
        return await _persist_item(state, draft)

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