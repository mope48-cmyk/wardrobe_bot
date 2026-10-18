"""Узлы графа LangGraph."""

import json
import logging
from html import escape as esc

from langchain_core.messages import AIMessage

from app.db.repository import (
    add_item,
    delete_item,
    get_colors,
    get_item,
    get_items,
    update_item_field,
)
from app.graph.state import BotState
from app.services.llm_stylist import generate_vasiliev_comment
from app.services.outfit import select_outfit, OutfitResult
from app.services.product_photo_bg import (
    is_generating,
    mark_generating,
    set_current_card,
)
from app.services.vision import classify_clothing
from app.services.weather import geocode_city, get_weather

logger = logging.getLogger(__name__)


# ---------- Клавиатуры ----------

# Спека клавиатуры = список рядов, каждый ряд — список подписей кнопок.
# Шага awaiting_confirm_category здесь нет: его клавиатура динамическая,
# зависит от вариантов распознавания.
STEP_KEYBOARDS: dict[str, list[list[str]]] = {
    "awaiting_photo": [["❌ Отмена"]],
    "awaiting_category": [["верх", "низ"], ["обувь", "аксессуар"], ["❌ Отмена"]],
    "awaiting_type": [["❌ Отмена"]],
    "awaiting_color": [["✅ Подтвердить", "🔄 Изменить"], ["❌ Отмена"]],
    "awaiting_color_text": [["❌ Отмена"]],
    "awaiting_layer": [["Базовый", "Средний"], ["Верхний"], ["❌ Отмена"]],
    "awaiting_warmth": [["1", "2", "3"], ["4", "5"], ["❌ Отмена"]],
    "awaiting_waterproof": [["да", "нет"], ["❌ Отмена"]],
    "awaiting_formal": [["casual", "business", "sport"], ["❌ Отмена"]],
    "awaiting_season": [["лето", "демисезон"], ["зима", "универсальная"], ["❌ Отмена"]],
    "awaiting_location": [["❌ Отмена"]],
    "awaiting_material": [
        ["Хлопок", "Шерсть"],
        ["Кожа", "Деним"],
        ["Трикотаж", "Синтетика"],
        ["Другое", "❌ Отмена"],
    ],
}

OCCASION_KEYBOARD = [["работа", "прогулка"], ["спорт", "встреча"], ["другое"], ["❌ Отмена"]]

MATERIAL_KEYBOARD = [
    ["Хлопок", "Шерсть"],
    ["Кожа", "Деним"],
    ["Трикотаж", "Синтетика"],
    ["Другое", "❌ Отмена"],
]

REMOVE_KB = "remove"

# Главное меню. Используется после /start, /cancel, сохранения вещи и т.п.
# Подписи на русском с эмодзи. Маршрутизация по ним — в builder.route_condition.
MENU_KEYBOARD = [
    ["👕 Мой гардероб", "🌤 Подобрать образ"],
    ["❓ Справка", "❌ Отмена"],
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
    rows.append(["❌ Отмена"])
    return rows


async def _persist_item(state: BotState, draft: dict) -> dict:
    """Сохранить вещь в БД и вернуть готовый ответ для пользователя.

    Используется и после подтверждения распознавания,
    и после ручного ввода.
    """
    item_colors = draft.get("colors") or []
    if isinstance(item_colors, str):
        item_colors = [item_colors]

    try:
        await add_item(
            user_id=state["user_id"],
            photo_file_id=draft["photo_file_id"],
            category=draft.get("category", "верх"),  # страховка
            type=draft.get("type", "вещь"),           # страховка
            colors=item_colors,
            material=draft.get("material", ""),
            # Безопасные значения по умолчанию:
            warmth_level=draft.get("warmth_level", 3),
            waterproof=draft.get("waterproof", False),
            formal_level=draft.get("formal_level", "casual"),
            season=draft.get("season", "универсальная"),
            layer=draft.get("layer", "base"),
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

    waterproof_str = "да" if draft.get("waterproof", False) else "нет"
    return {
        "messages": [
            AIMessage(
                content=(
                    "Готово! Сохранил вещь:\n"
                    f"• {draft.get('category', 'верх')} / {draft.get('type', 'вещь')}\n"
                    f"• цвета: {', '.join(item_colors) if item_colors else '—'}\n"
                    f"• материал: {draft.get('material') or '—'}\n"
                    f"• тепло: {draft.get('warmth_level', 3)}/5\n"
                    f"• водонепроницаемая: {waterproof_str}\n"
                    f"• стиль: {draft.get('formal_level', 'casual')}\n"
                    f"• сезон: {draft.get('season', 'универсальная')}\n\n"
                    "Добавьте ещё вещь командой /add."
                ),
                additional_kwargs=_kb(MENU_KEYBOARD),
            )
        ],
        "step": None, "draft_item": None, "intent": None,
    }


# ---------- Тексты ----------

WELCOME_TEXT = (
    "Ах, Боже мой! Наконец-то вы здесь! Я — ваш личный историк моды и стилист, "
    "и я просто счастлив приветствовать вас в моей цифровой гардеробной!\n\n"
    "Позвольте мне провести для вас небольшую экскурсию по моим владениям:\n\n"
    "<b>📸 Пополнение коллекции</b>\n"
    "Просто пришлите мне фотографию вашей вещи — я, как истинный ценитель, "
    "мгновенно распознаю её, определю цвет, материал и слой, и бережно помещу "
    "в ваш цифровой гардероб. Никаких лишних кнопок, только искусство!\n\n"
    "<b>👕 Инспекция гардероба (/list)</b>\n"
    "Нажмите на кнопку или введите команду, чтобы увидеть ваши сокровища. "
    "Листая карточки, вы сможете редактировать детали, безжалостно удалять "
    "ненужное или создать идеальное фото для каталога (на белом фоне, как в Vogue!).\n\n"
    "<b>🌤 Создание образа (/outfit)</b>\n"
    "Погода за окном — это катастрофа или божественный подарок? Отправьте мне "
    "геолокацию (или напишите город), скажите повод, и я лично подберу вам "
    "идеальный ансамбль, достойный обложки модного журнала!\n\n"
    "<b>❓ Справка (/info)</b>\n"
    "Если вы забыли, как управлять этим великолепием, я всегда готов напомнить.\n\n"
    "<b>❌ Отмена (/cancel)</b>\n"
    "Если всё пошло не по плану, и вы сбились с пути — просто нажмите эту кнопку, "
    "и мы начнём наш выход в свет заново.\n\n"
    "Творите моду, дорогие мои, и помните: безвкусица — это грех!"
)

HELP_TEXT = (
    "Ах, мои дорогие! Вы заглянули в мою скромную, но невероятно элегантную гардеробную! "
    "Позвольте мне провести для вас небольшую экскурсию.\n\n"
    "<b>📸 Пополнение коллекции</b>\n"
    "Просто отправьте мне фотографию вашей вещи. Я, как истинный ценитель, мгновенно "
    "распознаю её, определю цвет и материал, и бережно помещу в ваш цифровой гардероб.\n\n"
    "<b>👕 Инспекция гардероба (/list)</b>\n"
    "Нажмите на кнопку или введите команду, чтобы увидеть ваши сокровища. "
    "Листая карточки, вы сможете редактировать детали, безжалостно удалять ненужное "
    "или создать идеальное фото для каталога.\n\n"
    "<b>🌤 Создание образа (/outfit)</b>\n"
    "Погода за окном — это катастрофа или божественный подарок? Отправьте мне "
    "геолокацию (или напишите город), скажите повод, и я лично подберу вам идеальный "
    "ансамбль, достойный обложки Vogue!\n\n"
    "<b>❌ Отмена (/cancel)</b>\n"
    "Если всё пошло не по плану, и вы сбились с пути — просто нажмите эту кнопку, "
    "и мы начнём наш выход в свет заново.\n\n"
    "Творите моду, дорогие мои, и помните: безвкусица — это грех!"
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
                content="Великолепно! Всё отменено, как неудачный показ в Милане. Что мы будем делать дальше, мои дорогие?",
                additional_kwargs=_kb(MENU_KEYBOARD),
            )
        ],
        "intent": None, "step": None, "draft_item": None,
    }


async def fallback_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content="Это просто катастрофа! Я вас совершенно не понимаю. Наберите /info, чтобы я напомнил вам правила хорошего тона.")]}


async def stub_node(state: BotState) -> dict:
    return {"messages": [AIMessage(content="Эта команда появится позже.")]}


# ---------- /list: постраничный просмотр ----------

LIST_PREVIEW_LEN = 60  # сколько символов типа показывать в заголовке


COLOR_EMOJI: dict[str, str] = {
    "красный": "🔴", "бордовый": "🔴", "розовый": "🩷",
    "оранжевый": "🟠", "терракотовый": "🟠",
    "жёлтый": "🟡", "горчичный": "🟡",
    "зелёный": "🟢", "оливковый": "🟢", "хаки": "🟢", "мятный": "🟢",
    "голубой": "🔵", "бирюзовый": "🔵",
    "синий": "🔵", "тёмно-синий": "🔵",
    "фиолетовый": "🟣", "сиреневый": "🟣", "лиловый": "🟣",
    "коричневый": "🟤", "тёмно-коричневый": "🟤",
    "белый": "⚪", "кремовый": "⚪", "бежевый": "⚪",
    "чёрный": "⚫",
    "серый": "🩶", "тёмно-серый": "🩶", "светло-серый": "🩶",
}


def _format_color_with_emoji(color: str) -> str:
    emoji = COLOR_EMOJI.get(color.lower(), "🎨")
    return f"{emoji} {color}"


def _item_caption(item) -> str:
    """Подпись под фото вещи (компактный формат)."""
    parts = []

    # Цвет
    item_colors = get_colors(item)
    if item_colors:
        parts.append(f"цвет: {', '.join(item_colors)}")

    # Материал
    if item.material:
        parts.append(f"материал: {item.material}")

    # Слой
    layer_map = {
        "base": "базовый",
        "mid": "средний",
        "outer": "верхний",
        "none": "нет",
    }
    layer_ru = layer_map.get(getattr(item, "layer", "base"), "базовый")
    parts.append(f"слой: {layer_ru}")

    # Собираем атрибуты в одну строку через разделитель
    attrs_str = " | ".join(parts)

    waterproof_mark = "💧 " if item.waterproof else ""

    return (
        f"<b>{item.category} / {item.type}</b>\n"
        f"{attrs_str}\n"
        f"{waterproof_mark}тепло: {item.warmth_level}/5 | "
        f"стиль: {item.formal_level} | сезон: {item.season}"
    )


def _list_keyboard(
    index: int,
    total: int,
    has_generated: bool,
    show_original: bool,
    is_gen_now: bool,
) -> list[list[dict]]:
    """Inline-клавиатура карточки вещи."""
    prev_cb = "list:prev" if index > 0 else "list:noop:first"
    next_cb = "list:next" if index < total - 1 else "list:noop:last"

    if is_gen_now:
        row2 = [
            {"text": "⏳ Готовится…", "callback_data": "list:noop_generating"},
        ]
    elif has_generated:
        toggle_text = (
            "📷 Оригинал" if not show_original else "📸 Фото каталога"
        )
        row2 = [
            {"text": toggle_text, "callback_data": "list:toggle_photo"},
            {
                "text": "🗑 Удалить фото каталога",
                "callback_data": "list:delete_product_photo",
            },
        ]
    else:
        row2 = [
            {
                "text": "📸 Сделать фото для каталога",
                "callback_data": "list:make_product_photo",
            },
        ]

    return [
        [
            {"text": "✏️ Редактировать", "callback_data": "list:edit"},
            {"text": "🗑 Удалить вещь", "callback_data": "list:delete"},
        ],
        row2,
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

    show_original = bool(state.get("show_original") or False)
    has_generated = bool(item.generated_photo_file_id)
    is_gen = is_generating(item.id)

    # Запоминаем текущую вещь — нужно для фоновой задачи
    chat_id = state.get("chat_id")
    if chat_id:
        set_current_card(chat_id, item.id)

    if is_gen:
        photo_to_show = item.photo_file_id
        mode_mark = "\n\n⏳ <i>Готовится фото для каталога…</i>"
    elif has_generated and not show_original:
        photo_to_show = item.generated_photo_file_id
        mode_mark = "\n📸 <i>фото из каталога</i>"
    else:
        photo_to_show = item.photo_file_id
        mode_mark = ""

    header = f"📋 Вещь <b>{index + 1}</b> из <b>{total}</b>\n\n"
    caption = header + _item_caption(item) + mode_mark

    additional: dict = {
        "photo_file_id": photo_to_show,
        "inline_keyboard": _list_keyboard(
            index, total, has_generated, show_original, is_gen
        ),
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
        "show_original": show_original,
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
            "show_original": False,
        }

    ids = [item.id for item in items]
    return await _render_current({
        **state,
        "list_ids": ids,
        "list_index": 0,
        "list_source": "list",
        "edit_message_id": None,
        "show_original": False,
    })


async def list_prev_node(state: BotState) -> dict:
    """Перейти к предыдущей вещи. Сброс show_original."""
    index = state.get("list_index") or 0
    return await _render_current({
        **state,
        "list_index": max(0, index - 1),
        "show_original": False,
    })


async def list_next_node(state: BotState) -> dict:
    """Перейти к следующей вещи. Сброс show_original."""
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    return await _render_current({
        **state,
        "list_index": min(len(ids) - 1, index + 1),
        "show_original": False,
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
    "colors": "colors",
    "material": "material",
    "warmth": "warmth_level",
    "waterproof": "waterproof",
    "formal": "formal_level",
    "season": "season",
    "layer": "layer",
}

# Человекочитаемое имя поля
EDIT_FIELD_LABEL = {
    "category": "Категория",
    "type": "Тип",
    "colors": "Цвета",
    "material": "Материал",
    "warmth": "Тепло (1–5)",
    "waterproof": "Водонепроницаемость (да/нет)",
    "formal": "Стиль (casual / business / sport)",
    "season": "Сезон (лето / демисезон / зима / универсальная)",
    "layer": "Слой",
}

# Поля со свободным текстовым вводом
TEXT_EDIT_FIELDS = {"type", "colors"}

# Поля с фиксированными вариантами.
# Формат: field → [(подпись на кнопке, код для callback, значение для БД), ...]
CHOICE_EDIT_FIELDS: dict[str, list[tuple[str, str, object]]] = {
    "material": [
        ("Хлопок", "cotton", "хлопок"),
        ("Шерсть", "wool", "шерсть"),
        ("Кожа", "leather", "кожа"),
        ("Деним", "denim", "деним"),
        ("Трикотаж", "knit", "трикотаж"),
        ("Синтетика", "synthetic", "синтетика"),
    ],
    "category": [
        ("Верх", "top", "верх"),
        ("Низ", "bottom", "низ"),
        ("Обувь", "shoes", "обувь"),
        ("Аксессуар", "accessory", "аксессуар"),
    ],
    "warmth": [
        ("1", "1", 1),
        ("2", "2", 2),
        ("3", "3", 3),
        ("4", "4", 4),
        ("5", "5", 5),
    ],
    "waterproof": [
        ("Да", "yes", True),
        ("Нет", "no", False),
    ],
    "formal": [
        ("Casual", "casual", "casual"),
        ("Business", "business", "business"),
        ("Sport", "sport", "sport"),
    ],
    "season": [
        ("Лето", "summer", "лето"),
        ("Демисезон", "demi", "демисезон"),
        ("Зима", "winter", "зима"),
        ("Универсальная", "universal", "универсальная"),
    ],
    "layer": [
        ("Базовый", "base", "base"),
        ("Средний", "mid", "mid"),
        ("Верхний", "outer", "outer"),
        ("Нет", "none", "none"),
    ],
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

    if field == "colors":
        parts = [p.strip().lower() for p in text.split(",") if p.strip()]
        if 1 <= len(parts) <= 3 and all(len(p) >= 2 for p in parts):
            return True, parts, ""
        return False, None, (
            "Введите от 1 до 3 цветов через запятую."
        )

    if field == "material":
        if lower in VALID_MATERIALS:
            return True, lower, ""
        # Разрешаем и свободный ввод (лён, кашемир, вискоза и т.д.)
        if len(text) >= 2:
            return True, text, ""
        return False, None, "Напишите материал текстом."

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

    if field == "layer":
        layer_map = {
            "базовый": "base",
            "средний": "mid",
            "верхний": "outer",
            "нет": "none",
        }
        if lower in layer_map:
            return True, layer_map[lower], ""
        if lower in ("base", "mid", "outer", "none"):
            return True, lower, ""
        return False, None, "Выберите: базовый, средний, верхний или нет."

    return False, None, "Неизвестное поле."


def _field_value_str(item, field: str) -> str:
    """Текущее значение поля в виде строки для отображения."""
    if field == "colors":
        return ", ".join(get_colors(item)) or "—"

    if field == "layer":
        return {"base": "базовый", "mid": "средний", "outer": "верхний", "none": "нет"}.get(
            getattr(item, "layer", "base"), "базовый"
        )

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

    item_colors_str = ", ".join(get_colors(item)) or "—"
    caption = (
        "✏️ <b>Редактирование вещи</b>\n\n"
        f"{item.category} / {item.type}\n"
        f"цвета: {item_colors_str}\n"
        f"материал: {item.material or '—'}\n"
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
            {"text": "Цвета", "callback_data": "list:edit:colors"},
            {"text": "Материал", "callback_data": "list:edit:material"},
        ],
        [
            {"text": "Слой", "callback_data": "list:edit:layer"},
            {"text": "Сезон", "callback_data": "list:edit:season"},
        ],
        [
            {"text": "Тепло", "callback_data": "list:edit:warmth"},
            {"text": "Стиль", "callback_data": "list:edit:formal"},
        ],
        [
            {"text": "Водонепроницаемость", "callback_data": "list:edit:waterproof"},
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
    """Пользователь выбрал поле.

    Если поле из CHOICE_EDIT_FIELDS — показываем кнопки с вариантами.
    Если поле из TEXT_EDIT_FIELDS — просим ввести текстом.
    """
    text = (state.get("input_text") or "").strip()
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

    label = EDIT_FIELD_LABEL[field]
    current_str = _field_value_str(item, field)

    if field in CHOICE_EDIT_FIELDS:
        # Кнопки с вариантами
        buttons = CHOICE_EDIT_FIELDS[field]
        rows: list[list[dict]] = []
        row: list[dict] = []
        for btn_label, code, _ in buttons:
            row.append({
                "text": btn_label,
                "callback_data": f"list:set:{field}:{code}",
            })
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        rows.append([{"text": "❌ Отмена", "callback_data": "list:edit:cancel"}])

        if field == "material":
            hint = "Выберите кнопкой или напишите текстом (например: лён, кашемир):"
        else:
            hint = "Выберите новое значение:"

        caption = (
            f"✏️ <b>{label}</b>\n\n"
            f"Текущее значение: <b>{current_str}</b>\n\n"
            f"{hint}"
        )
        next_step = "list_edit_choice"
    else:
        # Текстовый ввод (тип, цвета)
        rows = [[{"text": "❌ Отмена", "callback_data": "list:edit:cancel"}]]
        if field == "colors":
            hint_text = "Введите от 1 до 3 цветов через запятую, например: красный, белый, синий."
        else:
            hint_text = "Введите новое значение текстом."
        caption = (
            f"✏️ <b>{label}</b>\n\n"
            f"Текущее значение: <b>{current_str}</b>\n\n"
            f"{hint_text}"
        )
        next_step = "list_edit_input"

    additional: dict = {
        "photo_file_id": item.photo_file_id,
        "inline_keyboard": rows,
        "edit_mode": "caption",
    }
    if edit_message_id is not None:
        additional["edit_message_id"] = edit_message_id

    return {
        "messages": [AIMessage(content=caption, additional_kwargs=additional)],
        "step": next_step,
        "intent": "list",
        "list_ids": ids,
        "list_index": index,
        "edit_message_id": edit_message_id,
        "draft_item": {"edit_field": field},
    }


async def list_edit_choice_node(state: BotState) -> dict:
    """Применить выбранное значение из кнопки.

    Приходит callback вида "list:set:category:top".
    Проверка значения не нужна: мы сами задали список вариантов.
    """
    text = (state.get("input_text") or "").strip()
    # "list:set:category:top" → ["list", "set", "category", "top"]
    parts = text.split(":", 3)
    if len(parts) != 4:
        return {
            "messages": [AIMessage(content="Ошибка выбора. Попробуйте снова.")],
            "step": None, "intent": None,
        }
    _, _, field, code = parts

    if field not in CHOICE_EDIT_FIELDS:
        return {
            "messages": [AIMessage(content="Неизвестное поле.")],
            "step": None, "intent": None,
        }

    # Ищем значение по коду
    value = None
    for _, c, v in CHOICE_EDIT_FIELDS[field]:
        if c == code:
            value = v
            break

    if value is None:
        return {
            "messages": [AIMessage(content="Неверный выбор. Попробуйте снова.")],
            "step": None, "intent": None,
        }

    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    user_id = state["user_id"]

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего редактировать.")],
            "step": None, "intent": None,
        }

    item_id = ids[index]

    # Для colors сериализуем в JSON (см. update_item_field — хранит строку)
    if field == "colors":
        value = json.dumps(value, ensure_ascii=False)

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

    # Для colors сериализуем в JSON
    if field == "colors":
        value = json.dumps(value, ensure_ascii=False)

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
        f"цвета: {', '.join(get_colors(item)) or '—'}\n\n"
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
        "show_original": False,
    })


async def list_delete_cancel_node(state: BotState) -> dict:
    """Отменить удаление — вернуться к просмотру текущей вещи.

    edit_mode="caption": фото то же, меняется только caption и кнопки.
    """
    return await _render_current(state, edit_mode="caption")


async def list_toggle_photo_node(state: BotState) -> dict:
    """Переключить: оригинал ↔ фото каталога.

    edit_mode="media" — чтобы edit_message_media заменил фото,
    а не только caption.
    """
    current = bool(state.get("show_original") or False)
    return await _render_current(
        {**state, "show_original": not current},
        edit_mode="media",
    )


async def list_delete_product_photo_node(state: BotState) -> dict:
    """Удалить фото из каталога без подтверждения."""
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    user_id = state["user_id"]

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего удалять.")],
            "step": None, "intent": None,
        }

    item_id = ids[index]
    ok = await update_item_field(
        item_id, user_id, "generated_photo_file_id", None
    )
    if not ok:
        return {
            "messages": [
                AIMessage(
                    content="Не удалось удалить фото. Попробуйте позже.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }

    return await _render_current(
        {**state, "show_original": False},
        edit_mode="caption",
    )


async def list_make_product_photo_node(state: BotState) -> dict:
    """Поставить задачу на генерацию и показать карточку со статусом.

    Реальная генерация запускается в handlers как asyncio-задача.
    """
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0

    if not ids or index >= len(ids):
        return {
            "messages": [AIMessage(content="Нечего обрабатывать.")],
            "step": None, "intent": None,
        }

    item = await get_item(ids[index])
    if item is None:
        return {
            "messages": [AIMessage(content="Вещь не найдена.")],
            "step": None, "intent": None,
        }

    # Если уже генерируется — просто перерисовываем карточку
    if is_generating(item.id):
        return await _render_current(state, edit_mode="caption")

    # Помечаем сразу, чтобы рендер карточки увидел статус
    mark_generating(item.id)

    result = await _render_current(state, edit_mode="caption")

    # Добавляем маркер — handlers запустит фоновую задачу
    if result["messages"]:
        extra = dict(result["messages"][0].additional_kwargs or {})
        extra["start_product_photo"] = {
            "item_id": item.id,
            "user_id": state["user_id"],
            "photo_file_id": item.photo_file_id,
            "category": item.category,
            "item_type": item.type,
            "card_index": index,
            "card_total": len(ids),
        }
        result["messages"][0].additional_kwargs = extra

    return result


# ---------- /add ----------

VALID_CATEGORIES = {"верх", "низ", "обувь", "аксессуар"}
VALID_FORMAL = {"casual", "business", "sport"}
VALID_SEASON = {"лето", "демисезон", "зима", "универсальная"}
VALID_MATERIALS = {"хлопок", "шерсть", "кожа", "деним", "трикотаж", "синтетика"}
VALID_LAYERS = {"базовый": "base", "средний": "mid", "верхний": "outer"}

STEP_TO_FIELD = {
    "awaiting_category": "category",
    "awaiting_type": "type",
    "awaiting_color": "colors",
    "awaiting_color_text": "colors",
    "awaiting_warmth": "warmth_level",
    "awaiting_waterproof": "waterproof",
    "awaiting_formal": "formal_level",
    "awaiting_season": "season",
    "awaiting_material": "material",
    "awaiting_layer": "layer",
}

NEXT_STEP = {
    "awaiting_category": (
        "awaiting_type",
        "А какой у нас тип вещи? Напишите (например: футболка, джинсы, куртка):",
    ),
    "awaiting_type": (
        "awaiting_color_text",
        "Ах, цвет! Это же основа образа. Введите от 1 до 3 цветов через запятую.",
    ),
    "awaiting_color": (
        "awaiting_warmth",
        "Насколько это тепло? Оцените от 1 (легкомысленная вещица) до 5 (как шуба в Сибири):",
    ),
    "awaiting_color_text": (
        "awaiting_warmth",
        "Насколько это тепло? Оцените от 1 (легкомысленная вещица) до 5 (как шуба в Сибири):",
    ),
    "awaiting_warmth": (
        "awaiting_waterproof",
        "Боюсь спросить, а эта вещь промокнет под дождём?",
    ),
    "awaiting_waterproof": (
        "awaiting_formal",
        "Для какого случая? casual, business или sport?",
    ),
    "awaiting_formal": (
        "awaiting_season",
        "Для какого сезона? лето, демисезон, зима или универсальная?",
    ),
    "awaiting_season": (
        "awaiting_material",
        "Из чего же это создано? Выберите материал:",
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

    if step in ("awaiting_color", "awaiting_color_text"):
        parts = [p.strip().lower() for p in text.split(",") if p.strip()]
        if 1 <= len(parts) <= 3 and all(len(p) >= 2 for p in parts):
            return True, parts, ""
        return False, None, (
            "Введите от 1 до 3 цветов через запятую, например: "
            "красный, белый, синий."
        )

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

    if step == "awaiting_material":
        if lower in VALID_MATERIALS:
            return True, lower, ""
        return False, None, "Выберите материал кнопкой из списка."

    if step == "awaiting_layer":
        if lower in VALID_LAYERS:
            return True, VALID_LAYERS[lower], ""
        return False, None, "Выберите: базовый, средний или верхний."

    return False, None, "Неизвестный шаг."


async def add_start_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Ах, я жду вашу фотографию! Пришлите мне шедевр.",
                additional_kwargs=_kb_for_step("awaiting_photo"),
            )
        ],
        "intent": "add", "step": "awaiting_photo", "draft_item": {},
    }


async def add_expect_photo_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Дорогуша, я всё еще жду фотографию! Не томите меня.",
                additional_kwargs=_kb_for_step("awaiting_photo"),
            )
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
                        "Боже мой, я не могу это распознать! Это точно одежда?\n\n"
                        "Подскажите мне категорию:"
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
                        "Боже мой, я не могу это распознать! Это точно одежда?\n\n"
                        "Подскажите мне категорию:"
                    ),
                    additional_kwargs=_kb_for_step("awaiting_category"),
                )
            ],
            "step": "awaiting_category",
            "draft_item": draft,
        }

    options = result["options"]
    draft["options"] = options
    # result["detected_colors"] — 1–3 основных цвета.
    # Если по какой-то причине пусто — берём топ-1.
    draft["predicted_colors"] = (
        result.get("detected_colors") or [result["color"]]
    )
    draft["predicted_layer"] = result.get("layer", "base")
    draft["confidence"] = result["confidence"]

    # Формируем список вариантов в тексте
    lines = ["Я распознал вещь. Вот мои варианты. Выберите, что это за чудо:\n"]
    for i, opt in enumerate(options, start=1):
        pct = int(opt["confidence"] * 100)
        if i == 1:
            lines.append(f"{i}. <b>{opt['type']}</b> — {pct}%")
        else:
            lines.append(f"{i}. {opt['type']} — {pct}%")

    lines.append("")

    detected = draft.get("predicted_colors") or [result["color"]]
    if len(detected) > 1:
        colors_str = ", ".join(
            _format_color_with_emoji(c) for c in detected
        )
        lines.append(f"Цвета: {colors_str}.")
    elif len(detected) == 1:
        lines.append(f"Цвет: {_format_color_with_emoji(detected[0])}.")
    else:
        lines.append("Цвет: —")

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
    predicted = draft.get("predicted_colors") or []

    # Формируем caption с эмодзи-цветами
    if len(predicted) > 1:
        colors_line = ", ".join(
            _format_color_with_emoji(c) for c in predicted
        )
        prompt = (
            f"Похоже, вещь многоцветная.\n\n"
            f"Основные цвета: {colors_line}\n\n"
            "Всё верно? Нажмите <b>✅ Подтвердить</b>, "
            "или <b>🔄 Изменить</b>, чтобы ввести вручную."
        )
    elif len(predicted) == 1:
        prompt = (
            f"Похоже, вещь цвета: "
            f"{_format_color_with_emoji(predicted[0])}.\n\n"
            "Всё верно? Нажмите <b>✅ Подтвердить</b>, "
            "или <b>🔄 Изменить</b>, чтобы ввести вручную."
        )
    else:
        prompt = (
            "Какого цвета вещь? Введите от 1 до 3 цветов через запятую."
        )

    def _go_to_color(d: dict) -> dict:
        return {
            "messages": [
                AIMessage(
                    content=prompt,
                    additional_kwargs=_kb_for_step("awaiting_color"),
                )
            ],
            "step": "awaiting_color",
            "draft_item": d,
        }

    # 1. Пользователь выбрал один из предложенных типов
    for opt in options:
        if opt["type"].lower() == text:
            draft["category"] = opt["category"]
            draft["type"] = opt["type"]
            draft["warmth_level"] = opt["warmth_level"]
            draft["waterproof"] = opt["waterproof"]
            draft["formal_level"] = opt["formal_level"]
            draft["season"] = opt["season"]
            draft["predicted_layer"] = opt.get("layer", "base")
            return _go_to_color(draft)

    # 2. "да" → топ-1
    if text in ("да", "yes", "верно", "ага", "ok", "+"):
        if options:
            opt = options[0]
            draft["category"] = opt["category"]
            draft["type"] = opt["type"]
            draft["warmth_level"] = opt["warmth_level"]
            draft["waterproof"] = opt["waterproof"]
            draft["formal_level"] = opt["formal_level"]
            draft["season"] = opt["season"]
            draft["predicted_layer"] = opt.get("layer", "base")
            return _go_to_color(draft)

    # 3. "другое" / "нет" → ручной путь
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


async def _maybe_ask_layer(state: BotState, draft: dict) -> dict:
    """Спросить слой, если это верх; иначе — сразу к материалу."""
    category = draft.get("category", "")
    layer = draft.get("predicted_layer", "base")

    if category in ("низ", "обувь", "аксессуар"):
        draft["layer"] = "none"
        return {
            "messages": [
                AIMessage(
                    content="Из чего сделана вещь? Выберите материал:",
                    additional_kwargs=_kb_for_step("awaiting_material"),
                )
            ],
            "step": "awaiting_material",
            "draft_item": draft,
        }

    layer_ru = {
        "base": "базовый",
        "mid": "средний",
        "outer": "верхний",
    }.get(layer, "базовый")
    return {
        "messages": [
            AIMessage(
                content=(
                    "К какому слою относится вещь?\n"
                    f"Предположительно: <b>{layer_ru}</b>.\n\n"
                    "Выберите слой кнопкой:"
                ),
                additional_kwargs=_kb_for_step("awaiting_layer"),
            )
        ],
        "step": "awaiting_layer",
        "draft_item": draft,
    }


async def add_layer_node(state: BotState) -> dict:
    """Обработка выбора слоя (только для категории «верх»)."""
    text = (state.get("input_text") or "").strip().lower()
    draft = dict(state.get("draft_item") or {})

    if text not in VALID_LAYERS:
        return {
            "messages": [
                AIMessage(
                    content="Выберите: базовый, средний или верхний.",
                    additional_kwargs=_kb_for_step("awaiting_layer"),
                )
            ]
        }

    draft["layer"] = VALID_LAYERS[text]
    return {
        "messages": [
            AIMessage(
                content="Из чего сделана вещь? Выберите материал:",
                additional_kwargs=_kb_for_step("awaiting_material"),
            )
        ],
        "step": "awaiting_material",
        "draft_item": draft,
    }


async def add_color_node(state: BotState) -> dict:
    """Обработка шага выбора цвета с предсказаниями."""
    text = (state.get("input_text") or "").strip().lower()
    draft = dict(state.get("draft_item") or {})
    predicted = draft.get("predicted_colors") or []

    # Подтверждение предсказанных цветов
    if text in ("✅ подтвердить", "подтвердить", "да", "yes", "ok", "+"):
        draft["colors"] = predicted or []
        return await _maybe_ask_layer(state, draft)

    # Ручной ввод
    if text in ("🔄 изменить", "изменить", "нет", "no", "-"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Введите от 1 до 3 цветов через запятую, "
                        "например: красный, белый, синий."
                    ),
                    additional_kwargs=_kb_for_step("awaiting_color_text"),
                )
            ],
            "step": "awaiting_color_text",
            "draft_item": draft,
        }

    # Пользователь сразу ввёл цвета через запятую
    valid, value, error = _validate_answer("awaiting_color", text)
    if valid:
        draft["colors"] = value
        return await _maybe_ask_layer(state, draft)

    return {
        "messages": [
            AIMessage(
                content=(
                    "Нажмите <b>✅ Подтвердить</b>, "
                    "<b>🔄 Изменить</b> или напишите цвета через запятую."
                ),
                additional_kwargs=_kb_for_step("awaiting_color"),
            )
        ],
    }


async def add_color_text_node(state: BotState) -> dict:
    """Обработка текстового ввода цветов (через запятую)."""
    text = (state.get("input_text") or "").strip()
    draft = dict(state.get("draft_item") or {})

    valid, value, error = _validate_answer("awaiting_color_text", text)
    if not valid:
        return {"messages": [AIMessage(content=error)]}

    draft["colors"] = value
    return await _maybe_ask_layer(state, draft)


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
        return {
            "messages": [
                AIMessage(
                    content="Из чего сделана вещь? Выберите материал:",
                    additional_kwargs=_kb_for_step("awaiting_material"),
                )
            ],
            "step": "awaiting_material",
            "draft_item": draft,
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


async def add_material_node(state: BotState) -> dict:
    """Обработка выбора материала.

    Принимает:
      - нажатие кнопки из VALID_MATERIALS (хлопок, шерсть, ...);
      - нажатие «Другое» → переход в режим текстового ввода;
      - сразу текст (вельвет, лён, кашемир) — тоже сохраняется.
    """
    text = (state.get("input_text") or "").strip()
    lower = text.lower()
    draft = dict(state.get("draft_item") or {})

    # «Другое» — переход в текстовый ввод (с подсказкой)
    if lower == "другое":
        return {
            "messages": [
                AIMessage(
                    content="Введите материал текстом (например: лён, кашемир, вискоза):"
                )
            ],
            "step": "awaiting_material_text",
            "draft_item": draft,
        }

    # Кнопка из списка
    if lower in VALID_MATERIALS:
        draft["material"] = lower
        return await _persist_item(state, draft)

    # Свободный текст — принимаем как материал
    if len(text) >= 2:
        draft["material"] = text
        return await _persist_item(state, draft)

    # Слишком коротко — просим ещё раз
    return {
        "messages": [
            AIMessage(
                content=(
                    "Выберите материал кнопкой или напишите его текстом "
                    "(например: лён, кашемир):"
                ),
                additional_kwargs=_kb_for_step("awaiting_material"),
            )
        ]
    }


async def add_material_text_node(state: BotState) -> dict:
    """Свободный ввод материала."""
    text = (state.get("input_text") or "").strip()
    if len(text) < 2:
        return {
            "messages": [AIMessage(content="Слишком коротко. Напишите материал.")]
        }

    draft = dict(state.get("draft_item") or {})
    draft["material"] = text
    return await _persist_item(state, draft)


# ---------- /outfit ----------

VALID_OCCASIONS = {"работа", "прогулка", "спорт", "встреча", "другое"}


def _validate_llm_selection(
    item_ids: list[int], items: list, weather: dict
) -> list | None:
    """Проверить, что LLM выбрала валидный комплект.

    Возвращает список вещей или None, если выбор невалиден.
    Учитываем:
      - какие категории вообще есть в гардеробе (чтобы не требовать
        от LLM того, чего у пользователя нет);
      - требования по слоям для текущей температуры, но не больше,
        чем слоёв вообще есть в гардеробе;
      - цветовой бюджет (не больше MAX_COLOR_SCORE).
    """
    by_id = {item.id: item for item in items}

    selected: list = []
    seen: set[int] = set()
    for iid in item_ids:
        if iid in seen:
            continue
        if iid in by_id:
            selected.append(by_id[iid])
            seen.add(iid)

    if not selected:
        return None

    cats = {item.category for item in selected}
    wardrobe_cats = {item.category for item in items}

    # 1. Проверка обязательных категорий
    has_top = "верх" in cats
    has_bottom = "низ" in cats
    has_shoes = "обувь" in cats
    has_dress = any(
        item.type.lower() in ("платье", "сарафан", "комбинезон")
        for item in selected
    )

    if "верх" in wardrobe_cats and not has_top:
        logger.info("LLM-валидация: нет верха")
        return None
    if "низ" in wardrobe_cats and not has_bottom and not has_dress:
        logger.info("LLM-валидация: нет низа")
        return None
    if "обувь" in wardrobe_cats and not has_shoes:
        logger.info("LLM-валидация: нет обуви")
        return None

    # 3. Проверка иерархии слоёв
    top_items = [i for i in selected if i.category == "верх"]
    layers = {getattr(i, "layer", "base") for i in top_items}

    # Если выбран верхний слой (outer) — обязаны быть base и mid
    if "outer" in layers:
        if not {"base", "mid"}.issubset(layers):
            logger.info("LLM-валидация: верхний слой (outer) без базового (base) и среднего (mid)")
            return None

    # Если выбран средний слой (mid) — обязан быть base
    if "mid" in layers:
        if "base" not in layers:
            logger.info("LLM-валидация: средний слой (mid) без базового (base)")
            return None

    # 4. Проверка цветового бюджета
    from app.services.outfit import _color_score
    total_color_score = sum(_color_score(i) for i in selected)
    if total_color_score > 3.0:
        logger.info(
            "LLM-валидация: превышен цветовой бюджет (%.1f > 3.0)",
            total_color_score,
        )
        return None

    return selected


async def outfit_start_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "Где мы сегодня блистаем? Отправьте геолокацию "
                    "или напишите город."
                ),
                additional_kwargs=_kb_for_step("awaiting_location"),
            )
        ],
        "intent": "outfit",
        "step": "awaiting_location",
        "draft_item": None,
        "location": None,
        "occasion": None,
        "weather": None,
        "outfit": None,
        "outfit_options": None,
        "outfit_index": 0,
        "outfit_city": None,
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
                        ),
                        additional_kwargs=_kb_for_step("awaiting_location"),
                    )
                ],
            }
        location = geo
    else:
        return {
            "messages": [
                AIMessage(
                    content="Жду геолокацию или название города. Или нажмите Отмена.",
                    additional_kwargs=_kb_for_step("awaiting_location"),
                )
            ],
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    "Понял, записал. А какой повод для этого выхода в свет? "
                    "Выберите один из вариантов."
                ),
                additional_kwargs=_kb(OCCASION_KEYBOARD),
            )
        ],
        "step": "awaiting_occasion",
        "location": location,
    }


async def outfit_occasion_node(state: BotState) -> dict:
    """Обработка повода: получаем варианты, показываем сбалансированный."""
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
                    content=(
                        "Что-то пошло не так с местоположением. "
                        "Начните заново: /outfit."
                    ),
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
                    content=(
                        "Не удалось получить погоду. "
                        "Попробуйте позже или начните заново: /outfit."
                    ),
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

    # Получаем список вариантов
    outfits = select_outfit(list(items), weather, text)

    if not outfits or (not outfits[0].items and outfits[0].no_candidates):
        return {
            "messages": [
                AIMessage(
                    content="Не удалось собрать комплект.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None, "location": None,
            "weather": weather,
        }

    # Извлекаем только ID вещей из объектов, чтобы LangGraph смог сохранить состояние
    outfit_ids = [[i.id for i in o.items] for o in outfits]

    best_idx = min(1, len(outfits) - 1) if len(outfits) > 1 else 0
    best_outfit = outfits[best_idx]

    city_name = loc.get("city") or "вашем городе"

    # Формируем заголовок
    city_str = f" ({loc['city']})" if loc.get("city") else ""
    temp = weather["temp"]
    feels_like = weather.get("feels_like", temp)
    wind = weather.get("wind_speed", 0)

    weather_line = f"🌤 <b>Погода{esc(city_str)}</b>: {weather['description']}, {temp:.0f}°C"
    if abs(feels_like - temp) >= 2:
        weather_line += f" (ощущается как {feels_like:.0f}°C)"
    if wind > 3:
        weather_line += f", ветер {wind:.0f} м/с"

    # Генерируем комментарий Васильева
    comment = await generate_vasiliev_comment(
        weather, city_name, text, best_outfit.items, max_chars=350,
    )
    if not comment:
        comment = "Ах, дорогие мои! Этот образ — просто спасение!"

    # Собираем текст
    lines = [f"{weather_line}\n<b>Повод:</b> {text}\n\n💬 <i>{esc(comment)}</i>\n"]
    by_cat = {}
    for item in best_outfit.items:
        by_cat.setdefault(item.category, []).append(item)

    for cat, label in [("верх", "👕 Верх"), ("низ", "👖 Низ"), ("обувь", "👟 Обувь")]:
        if cat in by_cat:
            lines.append(f"<b>{label}:</b>")
            for item in by_cat[cat]:
                wp = " 💧" if item.waterproof else ""
                mat = f" ({item.material})" if item.material else ""
                layer = f" [{getattr(item, 'layer', 'base')}]" if cat == "верх" else ""
                lines.append(
                    f"• {item.type}{layer}{mat} — тепло {item.warmth_level}/5{wp}"
                )
            lines.append("")

    caption = "\n".join(lines).rstrip()

    photo_ids = [
        item.generated_photo_file_id or item.photo_file_id
        for item in best_outfit.items
    ]

    # Кнопки навигации (теплее/легче)
    kb: list[list[dict]] = []
    row: list[dict] = []
    if best_idx > 0:
        row.append({"text": "❄️ Чуть легче", "callback_data": "outfit:lighter"})
    if best_idx < len(outfits) - 1:
        row.append({"text": "🔥 Чуть теплее", "callback_data": "outfit:warmer"})
    if row:
        kb.append(row)

    return {
        "messages": [
            AIMessage(
                content=caption,
                additional_kwargs={
                    "media_group": photo_ids,
                    "inline_keyboard": kb,
                },
            ),
            AIMessage(
                content=(
                    "Если хотите другой вариант, нажмите кнопки выше. "
                    "Или /outfit заново."
                ),
                additional_kwargs=_kb(MENU_KEYBOARD),
            ),
        ],
        "step": None, "intent": None, "location": None,
        "outfit_options": outfit_ids,  # <-- Сохраняем только ID!
        "outfit_index": best_idx,
        "outfit_city": city_name,
        "occasion": text,
        "weather": weather,
    }


async def outfit_variant_node(state: BotState) -> dict:
    """Обработка нажатия кнопок 'Теплее' или 'Легче'."""
    text = (state.get("input_text") or "").strip()
    outfits_ids = state.get("outfit_options") or []
    idx = state.get("outfit_index") or 0

    if not outfits_ids:
        return {
            "messages": [
                AIMessage(
                    content="Данные устарели. Начните заново: /outfit",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ]
        }

    if text == "outfit:warmer" and idx < len(outfits_ids) - 1:
        idx += 1
    elif text == "outfit:lighter" and idx > 0:
        idx -= 1

    # Получаем актуальные вещи из БД по сохраненным ID
    all_items = await get_items(state["user_id"])
    items_map = {i.id: i for i in all_items}
    current_ids = outfits_ids[idx]
    best_outfit_items = [items_map[i] for i in current_ids if i in items_map]

    weather = state.get("weather") or {}
    city_name = state.get("outfit_city") or "вашем городе"
    occasion = state.get("occasion") or "прогулка"

    comment = await generate_vasiliev_comment(
        weather, city_name, occasion, best_outfit_items, max_chars=350,
    )
    if not comment:
        comment = "Ах, вот это другой разговор!"

    # Формируем текст
    lines = [f"💬 <i>{esc(comment)}</i>\n"]
    by_cat = {}
    for item in best_outfit_items:
        by_cat.setdefault(item.category, []).append(item)

    for cat, label in [("верх", "👕 Верх"), ("низ", "👖 Низ"), ("обувь", "👟 Обувь")]:
        if cat in by_cat:
            lines.append(f"<b>{label}:</b>")
            for item in by_cat[cat]:
                wp = " 💧" if item.waterproof else ""
                mat = f" ({item.material})" if item.material else ""
                layer = f" [{getattr(item, 'layer', 'base')}]" if cat == "верх" else ""
                lines.append(
                    f"• {item.type}{layer}{mat} — тепло {item.warmth_level}/5{wp}"
                )
            lines.append("")

    # Кнопки
    kb: list[list[dict]] = []
    row: list[dict] = []
    if idx > 0:
        row.append({"text": "❄️ Чуть легче", "callback_data": "outfit:lighter"})
    if idx < len(outfits_ids) - 1:
        row.append({"text": "🔥 Чуть теплее", "callback_data": "outfit:warmer"})
    if row:
        kb.append(row)

    return {
        "messages": [
            AIMessage(
                content="\n".join(lines).rstrip(),
                additional_kwargs={
                    "media_group": [
                        i.generated_photo_file_id or i.photo_file_id
                        for i in best_outfit_items
                    ],
                    "inline_keyboard": kb,
                    "edit_message_id": state.get("edit_message_id"),
                },
            )
        ],
        "outfit_index": idx,
    }