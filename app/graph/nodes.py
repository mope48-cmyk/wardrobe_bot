"""Узлы графа LangGraph."""

import json
import logging
from html import escape as esc
from datetime import datetime

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
from app.services.outfit import select_outfit, OutfitResult, _is_dress
from app.services.product_photo_bg import (
    is_generating,
    mark_generating,
    set_current_card,
)
from app.services.vision import classify_clothing
from app.services.vision_gigachat import analyze_clothing_with_gigachat
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
    "awaiting_verdict_photo": [["❌ Отмена"]],   # ← НОВОЕ
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

ACCESSORY_OCCASIONS_KB = [
    ["работа", "прогулка"],
    ["спорт", "встреча"],
    ["Готово", "❌ Отмена"],
]

REMOVE_KB = "remove"

# Главное меню. Используется после /start, /cancel, сохранения вещи и т.п.
# Подписи на русском с эмодзи. Маршрутизация по ним — в builder.route_condition.
MENU_KEYBOARD = [
    ["👕 Мой гардероб", "🌤 Подобрать образ"],
    ["🧬 Мой стиль", "⚖️ Приговор"],
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
            accessory_occasions=draft.get("accessory_occasions", []),
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
        "album_queue": state.get("album_queue"),
    }


async def _persist_or_ask_accessory(
    state: BotState, draft: dict
) -> dict:
    """Сохранить вещь или, если это аксессуар, спросить поводы.

    Для не-аксессуаров — сразу вызывает _persist_item.
    Для аксессуаров — инициализирует пустой список поводов и
    переводит диалог в шаг awaiting_accessory_occasion.
    """
    if draft.get("category") == "аксессуар":
        draft["accessory_occasions"] = []  # Инициализируем пустой список
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Для каких поводов подходит этот аксессуар? "
                        "Выберите один или несколько, затем нажмите «Готово»:"
                    ),
                    additional_kwargs=_kb(ACCESSORY_OCCASIONS_KB),
                )
            ],
            "step": "awaiting_accessory_occasion",
            "draft_item": draft,
        }
    return await _persist_item(state, draft)


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
    """Если бот не понял команду — генерируем ответ в стиле Васильева."""
    user_text = state.get("input_text") or ""

    # Импортируем функцию генерации
    from app.services.llm_stylist import generate_chat_response

    # Если пользователь написал что-то (не отправил пустое сообщение)
    if user_text.strip():
        response = await generate_chat_response(user_text)
        return {"messages": [AIMessage(content=response, additional_kwargs=_kb(MENU_KEYBOARD))]}

    # Запасной вариант, если пришел пустой текст или стикер
    return {
        "messages": [
            AIMessage(
                content="Ах, это просто катастрофа! Я вас совершенно не понимаю. Наберите /info, чтобы я напомнил вам правила хорошего тона.",
                additional_kwargs=_kb(MENU_KEYBOARD),
            )
        ]
    }


async def dna_start_node(state: BotState) -> dict:
    """Стилевой ДНК-тест: анализ гардероба + комментарий Васильева."""
    from app.services.style_dna import (
        MIN_ITEMS_FOR_DNA,
        analyze_wardrobe,
        profile_to_llm_brief,
    )
    from app.services.llm_stylist import generate_dna_comment

    items = await get_items(state["user_id"])

    if not items:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Боже мой, гардероб пуст! Пришлите мне хотя бы "
                        f"{MIN_ITEMS_FOR_DNA} вещей через /add — и я составлю "
                        "ваш стилевой портрет."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }

    if len(items) < MIN_ITEMS_FOR_DNA:
        need = MIN_ITEMS_FOR_DNA - len(items)
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Для ДНК-профиля нужно минимум {MIN_ITEMS_FOR_DNA} "
                        f"вещей. У вас {len(items)}. Пришлите ещё {need} — "
                        "и я разберу ваш стиль по косточкам!"
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }

    profile = analyze_wardrobe(list(items))
    brief = profile_to_llm_brief(profile)
    comment = await generate_dna_comment(brief, max_chars=400)

    if not comment:
        comment = "Браво! Ваш стиль уникален и требует лишь небольших штрихов."

    # Текст, который будет подписью к картинке
    text = (
        "🧬 <b>Ваш стилевой ДНК готов!</b>\n\n"
        f"💬 <i>{esc(comment)}</i>\n\n"
        "📸 Я подготовил для вас красивую карточку. Делитесь ею с друзьями!"
    )

    # Данные для Pillow
    card_data = {
        "total_items": profile.total_items,
        "dominant_style": profile.dominant_style,
        "top_colors": profile.top_colors,
        "warmth_label": profile.warmth_label,
        "gaps": profile.gaps,
    }

    return {
        "messages": [
            AIMessage(
                content=text,
                additional_kwargs={
                    "dna_card_data": card_data,
                    "inline_keyboard": [
                        [{"text": "👕 К гардеробу", "callback_data": "list:menu"}]
                    ],
                },
            )
        ],
        "step": None,
        "intent": None,
    }


# ---------- /verdict: «Приговор от Васильева» ----------

VERDICT_START_TEXT = (
    "Боже мой, вы готовы предстать перед моим судом? Пришлите фотографию "
    "в полный рост — и я вынесу вердикт вашему образу. Только честно!"
)

VERDICT_EXPECT_TEXT = (
    "Я всё ещё жду вашу фотографию в полный рост! Без неё приговора не будет."
)


async def verdict_start_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=VERDICT_START_TEXT,
                additional_kwargs=_kb_for_step("awaiting_verdict_photo"),
            )
        ],
        "intent": "verdict",
        "step": "awaiting_verdict_photo",
        "draft_item": None,
    }


async def verdict_expect_photo_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=VERDICT_EXPECT_TEXT,
                additional_kwargs=_kb_for_step("awaiting_verdict_photo"),
            )
        ],
    }


# Telegram: лимит 4096 символов на сообщение. Берём 3800 —
# запас под HTML-теги, заголовок и «<i>…</i>».
_TG_VERDICT_CHUNK = 3800


def _split_verdict_for_telegram(
    text: str, limit: int = _TG_VERDICT_CHUNK
) -> list[str]:
    """Разбить длинный текст на части <= limit.

    Стараемся резать по границе предложения (`. `, `! `, `? `, перевод строки),
    чтобы не рвать фразу пополам. Если границы нет — режем по символу.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
            window.rfind("\n"),
        )
        # Если нашли границу во второй половине окна — режем по ней.
        # Иначе — жёстко по limit (лучше так, чем промахнуться).
        if cut < limit // 2:
            cut = limit
        else:
            cut += 1  # захватываем точку/!/? после пробела
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()

    if rest:
        parts.append(rest)
    return parts


async def verdict_photo_node(state: BotState) -> dict:
    from app.services.verdict import generate_verdict_vasiliev

    image_bytes = state.get("input_photo_bytes")
    if not image_bytes:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Боже мой, я не могу разглядеть фото! Пришлите "
                        "ещё раз — лучше в полный рост."
                    ),
                    additional_kwargs=_kb_for_step("awaiting_verdict_photo"),
                )
            ],
        }

    verdict = await generate_verdict_vasiliev(image_bytes)
    if not verdict:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Катастрофа! Мой внутренний критик сегодня молчит. "
                        "Попробуйте позже или пришлите другое фото."
                    ),
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }

    chunks = _split_verdict_for_telegram(verdict)

    messages: list[AIMessage] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            content = (
                "⚖️ <b>Приговор от ИИ стилиста</b>\n\n"
                f"💬 <i>{esc(chunk)}</i>"
            )
        else:
            content = f"💬 <i>{esc(chunk)}</i>"

        # Меню на каждой части — чтобы не висела старая клавиатура
        # «❌ Отмена» с шага awaiting_verdict_photo.
        messages.append(
            AIMessage(content=content, additional_kwargs=_kb(MENU_KEYBOARD))
        )

    return {
        "messages": messages,
        "step": None, "intent": None,
    }


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


def _category_menu_keyboard() -> list[list[dict]]:
    """Клавиатура выбора раздела гардероба."""
    return [
        [
            {"text": "👕 Верх", "callback_data": "list:cat:верх"},
            {"text": "👖 Низ", "callback_data": "list:cat:низ"},
        ],
        [
            {"text": "👟 Обувь", "callback_data": "list:cat:обувь"},
            {"text": "🧢 Аксессуары", "callback_data": "list:cat:аксессуар"},
        ],
        [{"text": "📋 Все вещи", "callback_data": "list:cat:all"}],
    ]


def _list_keyboard(
    index: int,
    total: int,
    has_generated: bool,
    show_original: bool,
    is_gen_now: bool,
    show_more_menu: bool,
) -> list[list[dict]]:
    """Inline-клавиатура карточки вещи (динамическая)."""
    prev_cb = "list:prev" if index > 0 else "list:noop:first"
    next_cb = "list:next" if index < total - 1 else "list:noop:last"

    # --- БАЗОВОЕ МЕНЮ (2 ряда) ---
    if not show_more_menu:
        return [
            [
                {"text": "⬅️ Назад", "callback_data": prev_cb},
                {"text": "Вперёд ➡️", "callback_data": next_cb},
            ],
            [
                {"text": "📁 К разделам", "callback_data": "list:menu"},
                {"text": "⚙️ Дополнительно", "callback_data": "list:more_toggle"},
            ],
        ]

    # --- РАСШИРЕННОЕ МЕНЮ (показывается по кнопке) ---
    if is_gen_now:
        row2 = [{"text": "⏳ Готовится…", "callback_data": "list:noop_generating"}]
    elif has_generated:
        toggle_text = "📷 Оригинал" if not show_original else "📸 Каталог"
        row2 = [
            {"text": toggle_text, "callback_data": "list:toggle_photo"},
            {"text": "🗑 Удалить фото", "callback_data": "list:delete_product_photo"},
        ]
    else:
        row2 = [
            {"text": "📸 Сделать фото", "callback_data": "list:make_product_photo"},
        ]

    return [
        [
            {"text": "✏️ Редактировать", "callback_data": "list:edit"},
            {"text": "🧠 ИИ-Анализ", "callback_data": "list:ai_analyze"},
        ],
        row2,
        [
            {"text": "🗑 Удалить вещь", "callback_data": "list:delete"},
        ],
        [
            {"text": "⬅️ Назад", "callback_data": prev_cb},
            {"text": "Вперёд ➡️", "callback_data": next_cb},
        ],
        [
            {"text": "📁 К разделам", "callback_data": "list:menu"},
            {"text": "⬆️ Скрыть", "callback_data": "list:more_toggle"},
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

    show_more_menu = bool(state.get("show_more_menu") or False)

    header = f"📋 Вещь <b>{index + 1}</b> из <b>{total}</b>\n\n"
    caption = header + _item_caption(item) + mode_mark

    additional: dict = {
        "photo_file_id": photo_to_show,
        "inline_keyboard": _list_keyboard(
            index, total, has_generated, show_original, is_gen, show_more_menu
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
        "show_more_menu": show_more_menu,  # <-- сохраняем состояние
    }


async def list_node(state: BotState) -> dict:
    """/list — показываем меню разделов."""
    items = await get_items(state["user_id"])
    if not items:
        return {
            "messages": [
                AIMessage(
                    content="Ваш гардероб пока пуст.\n\nДобавьте первую вещь командой /add.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }
    return await list_menu_node(state)


async def list_menu_node(state: BotState) -> dict:
    """Показывает меню выбора категорий."""
    return {
        "messages": [
            AIMessage(
                content="Выберите раздел гардероба:",
                additional_kwargs={"inline_keyboard": _category_menu_keyboard()},
            )
        ],
        "step": "list_menu",
        "intent": "list",
        "list_ids": None,  # Сбрасываем текущий список
        "list_index": 0,
    }


async def list_category_node(state: BotState) -> dict:
    """Обрабатывает выбор категории из меню."""
    text = (state.get("input_text") or "").strip()
    # Из "list:cat:верх" достаем "верх"
    category = text.split(":")[-1]

    items = await get_items(state["user_id"])

    # Фильтруем, если выбрана не "all"
    if category != "all":
        items = [i for i in items if i.category == category]

    if not items:
        return {
            "messages": [
                AIMessage(
                    content="В этом разделе пока пусто.",
                    additional_kwargs={"inline_keyboard": _category_menu_keyboard()},
                )
            ],
            "step": "list_menu",
            "intent": "list",
        }

    ids = [item.id for item in items]
    return await _render_current({
        **state,
        "list_ids": ids,
        "list_index": 0,
        "list_filter": category,
        "edit_message_id": None,  # Отправляем новое сообщение с фото
        "show_original": False,
        "show_more_menu": False,  # <-- ДОБАВИТЬ
    })


async def list_prev_node(state: BotState) -> dict:
    """Перейти к предыдущей вещи. Сброс show_original."""
    index = state.get("list_index") or 0
    return await _render_current({
        **state,
        "list_index": max(0, index - 1),
        "show_original": False,
        "show_more_menu": False,  # <-- ДОБАВИТЬ
    })


async def list_next_node(state: BotState) -> dict:
    """Перейти к следующей вещи. Сброс show_original."""
    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    return await _render_current({
        **state,
        "list_index": min(len(ids) - 1, index + 1),
        "show_original": False,
        "show_more_menu": False,  # <-- ДОБАВИТЬ
    })


async def list_noop_node(state: BotState) -> dict:
    """Ничего не делать (границы списка).

    Handlers сам покажет alert через callback.answer — здесь просто
    возвращаем пустой ответ, чтобы граф не падал.
    """
    return {}


async def list_more_toggle_node(state: BotState) -> dict:
    """Переключить расширенное/базовое меню карточки."""
    current = bool(state.get("show_more_menu") or False)
    return await _render_current(
        {**state, "show_more_menu": not current},
        edit_mode="caption",
    )


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


async def list_ai_analyze_node(state: BotState) -> dict:
    """Обновить характеристики уже сохраненной вещи через GigaChat."""
    from app.services.telegram_files import download_photo_by_file_id
    from app.services.vision_gigachat import analyze_clothing_with_gigachat

    ids = state.get("list_ids") or []
    index = state.get("list_index") or 0
    user_id = state["user_id"]

    if not ids or index >= len(ids):
        return {"messages": [AIMessage(content="Нечего анализировать.")]}

    item_id = ids[index]
    item = await get_item(item_id)
    if not item:
        return {"messages": [AIMessage(content="Вещь не найдена.")]}

    # 1. Скачиваем оригинал фото
    image_bytes = await download_photo_by_file_id(item.photo_file_id)
    if not image_bytes:
        return {
            "messages": [
                AIMessage(
                    content="❌ Не удалось скачать фото для анализа.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None,
        }

    # 2. Отправляем в GigaChat
    ai_data = await analyze_clothing_with_gigachat(image_bytes)
    if not ai_data:
        # Возвращаемся к карточке, чтобы не ломать UX
        res = await _render_current(state, edit_mode="caption")
        if res.get("messages"):
            extra = dict(res["messages"][0].additional_kwargs or {})
            # Можно было бы показать alert, но мы просто тихо вернем карточку.
            # Handlers сам разберется с текстом.
        return res

    # 3. Обновляем поля в БД
    updates = {
        "category": ai_data.get("category"),
        "type": ai_data.get("type"),
        "colors": json.dumps(ai_data.get("colors", []), ensure_ascii=False),
        "material": ai_data.get("material"),
        "warmth_level": ai_data.get("warmth_level"),
        "waterproof": ai_data.get("waterproof"),
        "formal_level": ai_data.get("formal_level"),
        "season": ai_data.get("season"),
        "layer": ai_data.get("layer"),
    }

    for field, value in updates.items():
        if value is not None:
            await update_item_field(item_id, user_id, field, value)

    logger.info("ИИ обновил вещь id=%s", item_id)

    # 4. Перерисовываем карточку с новыми данными
    return await _render_current(state, edit_mode="caption")


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

    # 1. Пробуем GigaChat (если есть фото)
    if image_bytes:
        ai_data = await analyze_clothing_with_gigachat(image_bytes)
        if ai_data:
            # Успех! Заполняем черновик
            draft.update(ai_data)

            # Красивый текст для подтверждения
            colors_str = ", ".join(draft.get("colors", []))
            text = (
                "Браво! Я изучил вещь:\n\n"
                f"• <b>{draft.get('category')} / {draft.get('type')}</b>\n"
                f"• цвета: {colors_str}\n"
                f"• материал: {draft.get('material') or '—'}\n"
                f"• тепло: {draft.get('warmth_level')}/5\n"
                f"• стиль: {draft.get('formal_level')}\n"
                f"• сезон: {draft.get('season')}\n\n"
                "Всё верно?"
            )
            return {
                "messages": [
                    AIMessage(
                        content=text,
                        additional_kwargs={
                            "inline_keyboard": [
                                [
                                    {"text": "✅ Всё верно", "callback_data": "add:ai:save"},
                                    {"text": "✏️ Изменить", "callback_data": "add:ai:edit"},
                                ]
                            ]
                        },
                    )
                ],
                "step": "awaiting_ai_confirm",
                "draft_item": draft,
            }

    # 2. Fallback (старый SigLIP или ручной ввод)
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
                    content="Не удалось распознать. Подскажите категорию:",
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


async def add_confirm_ai_node(state: BotState) -> dict:
    """Обработка кнопок под карточкой от GigaChat."""
    text = (state.get("input_text") or "").strip()
    draft = state.get("draft_item") or {}

    if text == "add:ai:save":
        # Сохраняем и радуемся
        return await _persist_or_ask_accessory(state, draft)

    elif text == "add:ai:edit":
        # Сохраняем, чтобы получить ID, и сразу открываем редактор
        res = await _persist_item(state, draft)
        # Если _persist_item вернула ошибку (нет сообщений) — выходим
        if not res.get("messages"):
            return res

        # Нам нужен ID только что созданной вещи.
        # Проще всего попросить пользователя зайти в /list,
        # но мы можем сделать хак: перезапрашиваем последнюю вещь.
        # В текущей архитектуре проще отправить в /list.
        return {
            "messages": [
                AIMessage(
                    content="Вещь сохранена! Вы можете отредактировать её в разделе /list.",
                    additional_kwargs=_kb(MENU_KEYBOARD),
                )
            ],
            "step": None, "intent": None, "draft_item": None,
        }

    return {
        "messages": [AIMessage(content="Выберите вариант на клавиатуре ниже.")]
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
        return await _persist_or_ask_accessory(state, draft)

    # Свободный текст — принимаем как материал
    if len(text) >= 2:
        draft["material"] = text
        return await _persist_or_ask_accessory(state, draft)

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
    return await _persist_or_ask_accessory(state, draft)


async def add_accessory_occasion_node(state: BotState) -> dict:
    """Обработка выбора поводов для аксессуара.

    Пользователь нажимает кнопки-поводы, они накапливаются в
    draft["accessory_occasions"]. Когда нажимает «Готово» — вещь
    сохраняется. Если ничего не выбрано — ставим все 4 повода
    (универсальный аксессуар).
    """
    text = (state.get("input_text") or "").strip().lower()
    draft = dict(state.get("draft_item") or {})
    occasions = draft.get("accessory_occasions", [])

    if text in ("готово", "✅ готово"):
        if not occasions:
            # Если ничего не выбрали, ставим универсальный
            draft["accessory_occasions"] = [
                "прогулка", "работа", "встреча", "спорт"
            ]
        return await _persist_item(state, draft)

    valid_occasions = {"работа", "прогулка", "спорт", "встреча"}
    if text in valid_occasions:
        if text not in occasions:
            occasions.append(text)
            draft["accessory_occasions"] = occasions

        selected_str = ", ".join(occasions)
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Выбрано: <b>{selected_str}</b>.\n"
                        "Добавьте ещё или нажмите «Готово»:"
                    ),
                    additional_kwargs=_kb(ACCESSORY_OCCASIONS_KB),
                )
            ],
            "step": "awaiting_accessory_occasion",
            "draft_item": draft,
        }

    return {
        "messages": [
            AIMessage(
                content="Выберите повод из кнопок ниже или нажмите «Готово».",
                additional_kwargs=_kb(ACCESSORY_OCCASIONS_KB),
            )
        ]
    }


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
    has_dress = any(_is_dress(i) for i in selected)

    if "верх" in wardrobe_cats and not has_top:
        logger.info("LLM-валидация: нет верха")
        return None
    if "низ" in wardrobe_cats and not has_bottom and not has_dress:
        logger.info("LLM-валидация: нет низа (и нет платья)")
        return None
    if "обувь" in wardrobe_cats and not has_shoes:
        logger.info("LLM-валидация: нет обуви")
        return None

    # 3. Проверка иерархии слоёв
    top_items = [i for i in selected if i.category == "верх"]
    layers = {getattr(i, "layer", "base") for i in top_items}

    # Верхний слой (outer) требует наличия базового (base)
    if "outer" in layers and "base" not in layers:
        logger.info("LLM-валидация: верхний слой (outer) без базового (base)")
        return None

    # Средний слой (mid) требует наличия базового (base)
    if "mid" in layers and "base" not in layers:
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
        "outfit_relaxed": None,
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


def _format_weather_for_collage(weather: dict) -> str:
    """Короткая строка погоды для обложки."""
    if not weather:
        return ""

    desc = weather.get("description", "").capitalize()
    temp = weather.get("temp", 0)
    feels_like = weather.get("feels_like", temp)
    wind = weather.get("wind_speed", 0)

    parts = [f"{desc}, {temp:.0f}°C"]
    if abs(feels_like - temp) >= 2:
        parts.append(f"ощущается как {feels_like:.0f}°C")
    if wind > 3:
        parts.append(f"ветер {wind:.0f} м/с")

    return " • ".join(parts)


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

    # Извлекаем только ID вещей и флаги плохого стиля
    outfit_ids = [[i.id for i in o.items] for o in outfits]
    outfit_relaxed = [list(o.relaxed_categories) for o in outfits]

    best_idx = min(1, len(outfits) - 1) if len(outfits) > 1 else 0
    best_outfit = outfits[best_idx]

    # Для комментария оставляем "вашем городе", если нет нормального названия
    city_for_prompt = loc.get("city") or "вашем городе"

    # Для обложки: если нет города, выводим координаты
    city_display = loc.get("city")
    if not city_display:
        lat = loc.get("lat")
        lon = loc.get("lon")
        if lat is not None and lon is not None:
            city_display = f"{lat:.4f}, {lon:.4f}"
        else:
            city_display = ""

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

    # Генерируем комментарий Васильева с учётом флагов плохого стиля
    comment = await generate_vasiliev_comment(
        weather, city_for_prompt, text, best_outfit.items,
        relaxed_categories=best_outfit.relaxed_categories,
        max_chars=350,
    )
    if not comment:
        comment = "Ах, дорогие мои! Этот образ — просто спасение!"

    # Собираем текст
    lines = [f"{weather_line}\n<b>Повод:</b> {text}\n\n💬 <i>{esc(comment)}</i>\n"]
    relaxed_cats = set(best_outfit.relaxed_categories)

    by_cat = {}
    for item in best_outfit.items:
        by_cat.setdefault(item.category, []).append(item)

    for cat, label in [
        ("верх", "👕 Верх"),
        ("низ", "👖 Низ"),
        ("обувь", "👟 Обувь"),
        ("аксессуар", "🧢 Аксессуары"),
    ]:
        if cat in by_cat:
            lines.append(f"<b>{label}:</b>")
            for item in by_cat[cat]:
                wp = " 💧" if item.waterproof else ""
                mark = " ⚠️" if item.category in relaxed_cats else ""
                mat = f" ({item.material})" if item.material else ""
                layer = f" [{getattr(item, 'layer', 'base')}]" if cat == "верх" else ""
                lines.append(
                    f"• {item.type}{layer}{mat} — тепло {item.warmth_level}/5{wp}{mark}"
                )
            lines.append("")

    caption = "\n".join(lines).rstrip()

    photo_ids = [
        item.generated_photo_file_id or item.photo_file_id
        for item in best_outfit.items
    ]

    # Извлекаем ID для кнопок
    kb: list[list[dict]] = []
    row: list[dict] = []
    if best_idx > 0:
        row.append({"text": "❄️ Чуть легче", "callback_data": "outfit:lighter"})
    if best_idx < len(outfits) - 1:
        row.append({"text": "🔥 Чуть теплее", "callback_data": "outfit:warmer"})
    if row:
        kb.append(row)

    # --- VOGUE-КОЛЛАЖ ---
    labels = [item.type for item in best_outfit.items]
    vogue_cfg = {
        "photo_file_ids": photo_ids,
        "labels": labels,
        "city": city_display,
        "weather_str": _format_weather_for_collage(weather),
        "date_str": datetime.now().strftime("%d.%m.%Y"),
    }

    messages = [
        AIMessage(
            content=caption,
            additional_kwargs={
                "vogue_collage": vogue_cfg,
                "media_group": photo_ids,   # fallback, если коллаж не соберётся
            },
        ),
    ]

    if kb:
        messages.append(
            AIMessage(
                content="Хотите другой вариант? Нажмите кнопку ниже:",
                additional_kwargs={"inline_keyboard": kb},
            )
        )

    messages.append(
        AIMessage(
            content="Или выберите другое действие в меню:",
            additional_kwargs=_kb(MENU_KEYBOARD),
        )
    )

    return {
        "messages": messages,
        "step": None, "intent": None, "location": None,
        "outfit_options": outfit_ids,
        "outfit_index": best_idx,
        "outfit_city": city_display,
        "outfit_relaxed": outfit_relaxed,
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

    # Достаем сохраненные флаги для текущего варианта
    outfit_relaxed = state.get("outfit_relaxed") or []
    current_relaxed = outfit_relaxed[idx] if idx < len(outfit_relaxed) else []

    comment = await generate_vasiliev_comment(
        weather, city_name, occasion, best_outfit_items,
        relaxed_categories=current_relaxed,
        max_chars=350,
    )
    if not comment:
        comment = "Ах, вот это другой разговор!"

    # Формируем текст
    lines = [f"💬 <i>{esc(comment)}</i>\n"]
    relaxed_cats = set(current_relaxed)

    by_cat = {}
    for item in best_outfit_items:
        by_cat.setdefault(item.category, []).append(item)

    for cat, label in [
        ("верх", "👕 Верх"),
        ("низ", "👖 Низ"),
        ("обувь", "👟 Обувь"),
        ("аксессуар", "🧢 Аксессуары"),
    ]:
        if cat in by_cat:
            lines.append(f"<b>{label}:</b>")
            for item in by_cat[cat]:
                wp = " 💧" if item.waterproof else ""
                mark = " ⚠️" if item.category in relaxed_cats else ""
                mat = f" ({item.material})" if item.material else ""
                layer = f" [{getattr(item, 'layer', 'base')}]" if cat == "верх" else ""
                lines.append(
                    f"• {item.type}{layer}{mat} — тепло {item.warmth_level}/5{wp}{mark}"
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

    labels = [i.type for i in best_outfit_items]
    vogue_cfg = {
        "photo_file_ids": [
            i.generated_photo_file_id or i.photo_file_id
            for i in best_outfit_items
        ],
        "labels": labels,
        "city": state.get("outfit_city") or "",
        "weather_str": _format_weather_for_collage(weather),
        "date_str": datetime.now().strftime("%d.%m.%Y"),
    }

    messages = [
        AIMessage(
            content="\n".join(lines).rstrip(),
            additional_kwargs={
                "vogue_collage": vogue_cfg,
                "media_group": vogue_cfg["photo_file_ids"],  # fallback
            },
        )
    ]

    if kb:
        messages.append(
            AIMessage(
                content="Что-то ещё?",
                additional_kwargs={"inline_keyboard": kb},
            )
        )

    return {
        "messages": messages,
        "outfit_index": idx,
    }


# ============================================================
# /morning: Утренний образ
# ============================================================


async def morning_settings_node(state: BotState) -> dict:
    from app.db.repository import get_morning_settings

    settings = await get_morning_settings(state["user_id"])

    status = "Включена ✅" if settings.enabled else "Выключена ❌"
    days_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
    active_days = ", ".join(days_map[int(d)] for d in settings.days.split(",") if d) or "нет"

    text = (
        f"☀️ <b>Утренняя сводка</b>\n\n"
        f"Статус: {status}\n"
        f"Время: {settings.time_str}\n"
        f"Город: {settings.city or 'не указан'}\n"
        f"Повод: {settings.occasion}\n"
        f"Дни: {active_days}\n\n"
        f"Настройте параметры:"
    )

    kb = [
        [{"text": "🔄 Вкл/Выкл", "callback_data": "morning:toggle"}],
        [
            {"text": "⏰ Время", "callback_data": "morning:time"},
            {"text": "📍 Город", "callback_data": "morning:city"},
        ],
        [
            {"text": "👔 Повод", "callback_data": "morning:occasion"},
            {"text": "📅 Дни", "callback_data": "morning:days"},
        ],
        [{"text": "❌ Закрыть", "callback_data": "list:menu"}],
    ]

    return {
        "messages": [
            AIMessage(content=text, additional_kwargs={"inline_keyboard": kb})
        ],
        "step": "morning_menu",
    }


async def morning_toggle_node(state: BotState) -> dict:
    from app.db.repository import get_morning_settings, update_morning_settings

    settings = await get_morning_settings(state["user_id"])
    await update_morning_settings(state["user_id"], enabled=not settings.enabled)
    return await morning_settings_node(state)


async def morning_set_time_node(state: BotState) -> dict:
    import re

    text = (state.get("input_text") or "").strip()
    if not re.match(r"^\d{2}:\d{2}$", text):
        return {
            "messages": [
                AIMessage(
                    content="Неверный формат. Введите в формате ЧЧ:ММ (например, 08:00)."
                )
            ]
        }

    from app.db.repository import update_morning_settings

    await update_morning_settings(state["user_id"], time_str=text, enabled=True)
    return await morning_settings_node(state)


async def morning_set_city_node(state: BotState) -> dict:
    text = (state.get("input_text") or "").strip()
    if not text:
        return {"messages": [AIMessage(content="Введите название города.")]}

    from app.services.weather import geocode_city

    geo = await geocode_city(text)
    if not geo:
        return {
            "messages": [AIMessage(content="Город не найден. Попробуйте еще раз.")]
        }

    from app.db.repository import update_morning_settings

    await update_morning_settings(
        state["user_id"],
        city=geo.get("city", text),
        lat=geo["lat"],
        lon=geo["lon"],
    )
    return await morning_settings_node(state)


async def morning_set_occasion_node(state: BotState) -> dict:
    text = (state.get("input_text") or "").strip().lower()
    if text in ["работа", "прогулка", "спорт", "встреча", "другое"]:
        from app.db.repository import update_morning_settings

        await update_morning_settings(state["user_id"], occasion=text)
        return await morning_settings_node(state)
    return {
        "messages": [
            AIMessage(
                content="Выберите из кнопок ниже.",
                additional_kwargs=_kb(OCCASION_KEYBOARD),
            )
        ]
    }


async def morning_days_node(state: BotState) -> dict:
    text = (state.get("input_text") or "").strip()
    if text:
        try:
            days = [str(int(d)) for d in text.split(",") if 0 <= int(d) <= 6]
            from app.db.repository import update_morning_settings

            await update_morning_settings(state["user_id"], days=",".join(days))
            return await morning_settings_node(state)
        except Exception:
            pass
    return {
        "messages": [
            AIMessage(
                content=(
                    "Введите дни цифрами от 0 (Пн) до 6 (Вс) через запятую. "
                    "Например: 1,2,3,4,5"
                )
            )
        ]
    }


# Обертки для перехода в режим ввода (текст)
async def morning_ask_time_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(content="Введите время в формате ЧЧ:ММ (например, 08:00):")
        ],
        "step": "morning_await_time",
    }


async def morning_ask_city_node(state: BotState) -> dict:
    return {
        "messages": [AIMessage(content="Введите название города:")],
        "step": "morning_await_city",
    }


async def morning_ask_occasion_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Выберите повод:",
                additional_kwargs=_kb(OCCASION_KEYBOARD),
            )
        ],
        "step": "morning_await_occasion",
    }


async def morning_ask_days_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "Введите дни цифрами от 0 (Пн) до 6 (Вс) через запятую. "
                    "Например: 1,2,3,4,5"
                )
            )
        ],
        "step": "morning_await_days",
    }