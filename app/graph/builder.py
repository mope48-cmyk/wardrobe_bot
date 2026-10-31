"""Сборка графа LangGraph."""

import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import BotState

# Соответствие подписей кнопок главного меню → внутренние интенты.
# Должно совпадать с MENU_KEYBOARD из nodes.py.
MENU_LABELS: dict[str, str] = {
    "добавить вещь": "add_start",
    "мой гардероб": "list",
    "extra": "more_services",
    "подобрать образ": "outfit_start",
    "мой стиль": "dna_start",          # ← НОВОЕ
    "приговор": "verdict_start",       # ← НОВОЕ
    "помощь": "help",
    "справка": "help",
    "отмена": "cancel",
}


def _strip_emoji(text: str) -> str:
    """Убрать эмодзи и пунктуацию, оставить буквы и пробелы.

    "➕ Добавить вещь" → "Добавить вещь"
    "🌤 Подобрать образ" → "Подобрать образ"
    """
    cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё\s]", " ", text)
    return " ".join(cleaned.split())


def _match_menu_button(text: str) -> str | None:
    """Проверить, является ли текст одной из кнопок меню.

    Возвращает интент ("add_start" и т.п.) или None.
    Сравнение — точное (не подстрока), чтобы не путать с ответами
    в диалоге: например, пользователь может написать "помощь нужна"
    как произвольный текст — не должны на это реагировать.
    """
    cleaned = _strip_emoji(text).lower()
    return MENU_LABELS.get(cleaned)


def _match_free_text_intent(text: str) -> str | None:
    """Распознаем намерения пользователя по ключевым словам."""
    lower = text.lower()

    # Приговор от Васильева — первым, т.к. может пересекаться с "образ"
    if any(k in lower for k in [
        "приговор", "вердикт", "оцени меня", "оцени мой образ",
        "оцени образ", "что скажешь про образ",
    ]):
        return "verdict_start"

    # Стилевой ДНК-тест — проверяем первым, чтобы «покажи мой стиль»
    # не попадал в лист-интент из-за слова «покажи».
    if any(k in lower for k in ["днк", "мой стиль", "стилевой профиль", "профиль стиля"]):
        return "dna_start"

    # Интенты для гардероба
    if any(k in lower for k in ["гардероб", "вещи", "одежда", "покажи", "шкаф", "список", "куртки", "штаны", "посмотреть", "показать"]):
        return "list"

    # Интенты для подбора образа
    if any(k in lower for k in ["надеть", "образ", "подобрать", "подбери", "подберите", "одеть", "комплект", "выйти в свет", "прогулка", "собрать", "собери"]):
        return "outfit_start"

    # Интенты для отмены
    if any(k in lower for k in ["отмена", "отмени", "стоп", "хватит", "надоело", "отменить"]):
        return "cancel"

    # Интенты для помощи
    if any(k in lower for k in ["помощь", "справка", "умеешь", "команды", "инструкция", "помоги"]):
        return "help"

    return None


def route_condition(state: BotState) -> str:
    """Определить, в какой узел идти, на основе ввода и текущего шага."""
    text = (state.get("input_text") or "").strip()
    step = state.get("step")

    # 0. Callback'и из inline-кнопок
    if text == "list:prev": return "list_prev"
    if text == "list:next": return "list_next"
    if text.startswith("list:noop"): return "list_noop"
    if text == "list:edit": return "list_edit"
    if text == "list:edit:cancel": return "list_edit_cancel"
    if text.startswith("list:edit:"): return "list_edit_field"
    if text.startswith("list:set:"): return "list_edit_choice"
    if text == "list:delete": return "list_delete"
    if text == "list:delete_confirm": return "list_delete_do"
    if text == "list:delete_cancel": return "list_delete_cancel"
    if text == "list:toggle_photo": return "list_toggle_photo"
    if text == "list:delete_product_photo": return "list_delete_product_photo"
    if text == "list:make_product_photo": return "list_make_product_photo"
    if text == "list:ai_analyze": return "list_ai_analyze"
    if text == "list:more_toggle": return "list_more_toggle"
    if text == "list:menu": return "list_menu"
    if text.startswith("list:cat:"): return "list_category"
    if text == "outfit:warmer": return "outfit_variant"
    if text == "outfit:lighter": return "outfit_variant"
    if text in ("add:ai:save", "add:ai:edit"): return "add_confirm_ai"

    # Callback'и утренней сводки
    if text == "morning:toggle": return "morning_toggle"
    if text == "morning:time": return "morning_ask_time"
    if text == "morning:city": return "morning_ask_city"
    if text == "morning:occasion": return "morning_ask_occasion"
    if text == "morning:days": return "morning_ask_days"

    # Callback'и меню услуг
    if text == "menu:outfit": return "outfit_start"
    if text == "menu:dna": return "dna_start"
    if text == "menu:verdict": return "verdict_start"
    if text == "menu:morning": return "morning_settings"

    # 1. Slash-команды
    if text.startswith("/"):
        cmd = text.split()[0].lower()
        if cmd == "/start": return "start"
        if cmd in ("/help", "/info"): return "help"
        if cmd == "/cancel": return "cancel"
        if cmd == "/add": return "add_start"
        if cmd == "/list": return "list"
        if cmd == "/outfit": return "outfit_start"
        if cmd == "/dna": return "dna_start"
        if cmd == "/verdict": return "verdict_start"
        if cmd == "/morning": return "morning_settings"

    # 2. Кнопки главного меню
    menu_intent = _match_menu_button(text)
    if menu_intent:
        return menu_intent

    # 3. Интенты из свободного текста (если мы НЕ в середине активного диалога)
    active_dialogs = (
        "awaiting_photo",
        "awaiting_category",
        "awaiting_type",
        "awaiting_color",
        "awaiting_color_text",
        "awaiting_layer",
        "awaiting_warmth",
        "awaiting_waterproof",
        "awaiting_formal",
        "awaiting_season",
        "awaiting_material",
        "awaiting_material_text",
        "awaiting_location",
        "awaiting_occasion",
        "awaiting_verdict_photo",
        "awaiting_ai_confirm",
        "awaiting_confirm_category",
        "awaiting_accessory_occasion",
        "list_edit_input",
        "list_edit_choice",
        "morning_await_time",
        "morning_await_city",
        "morning_await_occasion",
        "morning_await_days",
        "morning_menu",
    )
    if step not in active_dialogs:
        free_intent = _match_free_text_intent(text)
        if free_intent:
            return free_intent

    # 4. Фото вне активного диалога
    if state.get("input_photo_file_id") and not step:
        return "add_photo"

    # 5. Текущий шаг диалога
    if step == "awaiting_photo":
        if state.get("input_photo_file_id"): return "add_photo"
        return "add_expect_photo"
    if step == "awaiting_confirm_category": return "add_confirm_category"
    if step == "awaiting_color": return "add_color"
    if step == "awaiting_color_text": return "add_color_text"
    if step == "awaiting_layer": return "add_layer"
    if step == "awaiting_location": return "outfit_location"
    if step == "awaiting_occasion": return "outfit_occasion"
    if step == "list_edit_input": return "list_edit_save"
    if step == "list_edit_choice" and not text.startswith("list:set:"): return "list_edit_save"
    if step == "awaiting_material": return "add_material"
    if step == "awaiting_material_text": return "add_material_text"
    if step == "awaiting_accessory_occasion": return "add_accessory_occasion"
    if step == "awaiting_verdict_photo":
        if state.get("input_photo_file_id"): return "verdict_photo"
        return "verdict_expect_photo"
    if step == "awaiting_ai_confirm": return "add_confirm_ai"
    if step == "morning_await_time": return "morning_set_time"
    if step == "morning_await_city": return "morning_set_city"
    if step == "morning_await_occasion": return "morning_set_occasion"
    if step == "morning_await_days": return "morning_days"
    if step and step.startswith("awaiting_"): return "add_attribute"

    return "fallback"


async def router_node(state: BotState) -> dict:
    return {}


def build_graph():
    builder = StateGraph(BotState)

    builder.add_node("router", router_node)
    builder.add_node("start", nodes.start_node)
    builder.add_node("help", nodes.help_node)
    builder.add_node("more_services", nodes.more_services_node)
    builder.add_node("cancel", nodes.cancel_node)
    builder.add_node("fallback", nodes.fallback_node)
    builder.add_node("stub", nodes.stub_node)

    # /list и навигация по списку
    builder.add_node("list", nodes.list_node)
    builder.add_node("list_menu", nodes.list_menu_node)
    builder.add_node("list_category", nodes.list_category_node)
    builder.add_node("list_prev", nodes.list_prev_node)
    builder.add_node("list_next", nodes.list_next_node)
    builder.add_node("list_noop", nodes.list_noop_node)
    builder.add_node("list_edit", nodes.list_edit_node)
    builder.add_node("list_edit_field", nodes.list_edit_field_node)
    builder.add_node("list_edit_choice", nodes.list_edit_choice_node)
    builder.add_node("list_edit_save", nodes.list_edit_save_node)
    builder.add_node("list_edit_cancel", nodes.list_edit_cancel_node)
    builder.add_node("list_delete", nodes.list_delete_node)
    builder.add_node("list_delete_do", nodes.list_delete_do_node)
    builder.add_node("list_delete_cancel", nodes.list_delete_cancel_node)
    builder.add_node("list_toggle_photo", nodes.list_toggle_photo_node)
    builder.add_node(
        "list_delete_product_photo", nodes.list_delete_product_photo_node
    )
    builder.add_node("list_make_product_photo", nodes.list_make_product_photo_node)
    builder.add_node("list_ai_analyze", nodes.list_ai_analyze_node)
    builder.add_node("list_more_toggle", nodes.list_more_toggle_node)

    # /outfit
    builder.add_node("outfit_start", nodes.outfit_start_node)
    builder.add_node("outfit_location", nodes.outfit_location_node)
    builder.add_node("outfit_occasion", nodes.outfit_occasion_node)
    builder.add_node("outfit_variant", nodes.outfit_variant_node)

    # /dna
    builder.add_node("dna_start", nodes.dna_start_node)

    # /verdict
    builder.add_node("verdict_start", nodes.verdict_start_node)
    builder.add_node("verdict_expect_photo", nodes.verdict_expect_photo_node)
    builder.add_node("verdict_photo", nodes.verdict_photo_node)

    # /add
    builder.add_node("add_start", nodes.add_start_node)
    builder.add_node("add_expect_photo", nodes.add_expect_photo_node)
    builder.add_node("add_photo", nodes.add_photo_node)
    builder.add_node("add_confirm_ai", nodes.add_confirm_ai_node)
    builder.add_node("add_confirm_category", nodes.add_confirm_category_node)
    builder.add_node("add_color", nodes.add_color_node)
    builder.add_node("add_color_text", nodes.add_color_text_node)
    builder.add_node("add_layer", nodes.add_layer_node)
    builder.add_node("add_attribute", nodes.add_attribute_node)
    builder.add_node("add_material", nodes.add_material_node)
    builder.add_node("add_material_text", nodes.add_material_text_node)
    builder.add_node("add_accessory_occasion", nodes.add_accessory_occasion_node)

    # /morning
    builder.add_node("morning_settings", nodes.morning_settings_node)
    builder.add_node("morning_toggle", nodes.morning_toggle_node)
    builder.add_node("morning_set_time", nodes.morning_set_time_node)
    builder.add_node("morning_set_city", nodes.morning_set_city_node)
    builder.add_node("morning_set_occasion", nodes.morning_set_occasion_node)
    builder.add_node("morning_days", nodes.morning_days_node)
    builder.add_node("morning_ask_time", nodes.morning_ask_time_node)
    builder.add_node("morning_ask_city", nodes.morning_ask_city_node)
    builder.add_node("morning_ask_occasion", nodes.morning_ask_occasion_node)
    builder.add_node("morning_ask_days", nodes.morning_ask_days_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_condition,
        {
            "start": "start", "help": "help", "cancel": "cancel",
            "fallback": "fallback", "stub": "stub", "list": "list",
            "more_services": "more_services",
            "list_menu": "list_menu",
            "list_category": "list_category",
            "list_prev": "list_prev",
            "list_next": "list_next",
            "list_noop": "list_noop",
            "list_edit": "list_edit",
            "list_edit_field": "list_edit_field",
            "list_edit_choice": "list_edit_choice",
            "list_edit_save": "list_edit_save",
            "list_edit_cancel": "list_edit_cancel",
            "list_delete": "list_delete",
            "list_delete_do": "list_delete_do",
            "list_delete_cancel": "list_delete_cancel",
            "list_toggle_photo": "list_toggle_photo",
            "list_delete_product_photo": "list_delete_product_photo",
            "list_make_product_photo": "list_make_product_photo",
            "list_ai_analyze": "list_ai_analyze",
            "list_more_toggle": "list_more_toggle",
            "outfit_start": "outfit_start",
            "outfit_location": "outfit_location",
            "outfit_occasion": "outfit_occasion",
            "outfit_variant": "outfit_variant",
            "dna_start": "dna_start",
            "verdict_start": "verdict_start",
            "verdict_expect_photo": "verdict_expect_photo",
            "verdict_photo": "verdict_photo",
            "add_start": "add_start",
            "add_expect_photo": "add_expect_photo",
            "add_photo": "add_photo",
            "add_confirm_ai": "add_confirm_ai",
            "add_confirm_category": "add_confirm_category",
            "add_color": "add_color",
            "add_color_text": "add_color_text",
            "add_layer": "add_layer",
            "add_attribute": "add_attribute",
            "add_material": "add_material",
            "add_material_text": "add_material_text",
            "add_accessory_occasion": "add_accessory_occasion",
            "morning_settings": "morning_settings",
            "morning_toggle": "morning_toggle",
            "morning_set_time": "morning_set_time",
            "morning_set_city": "morning_set_city",
            "morning_set_occasion": "morning_set_occasion",
            "morning_days": "morning_days",
            "morning_ask_time": "morning_ask_time",
            "morning_ask_city": "morning_ask_city",
            "morning_ask_occasion": "morning_ask_occasion",
            "morning_ask_days": "morning_ask_days",
        },
    )

    terminal_nodes = [
        "start", "help", "cancel", "fallback", "stub", "list",
        "more_services",
        "list_menu", "list_category",
        "list_prev", "list_next", "list_noop", "list_edit", "list_delete",
        "list_delete_do", "list_delete_cancel",
        "list_toggle_photo", "list_make_product_photo",
        "list_ai_analyze",
        "list_more_toggle",
        "list_delete_product_photo",
        "list_edit_field", "list_edit_choice", "list_edit_save", "list_edit_cancel",
        "outfit_start", "outfit_location", "outfit_occasion", "outfit_variant",
        "dna_start",
        "verdict_start", "verdict_expect_photo", "verdict_photo",
        "add_start", "add_expect_photo", "add_photo",
        "add_confirm_ai",
        "add_confirm_category", "add_attribute",
        "add_color", "add_color_text", "add_layer",
        "add_material", "add_material_text", "add_accessory_occasion",
        "morning_settings", "morning_toggle",
        "morning_set_time", "morning_set_city", "morning_set_occasion", "morning_days",
        "morning_ask_time", "morning_ask_city", "morning_ask_occasion", "morning_ask_days",
    ]
    for name in terminal_nodes:
        builder.add_edge(name, END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)