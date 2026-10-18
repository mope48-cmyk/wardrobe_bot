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
    "подобрать образ": "outfit_start",
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


def route_condition(state: BotState) -> str:
    """Определить, в какой узел идти, на основе ввода и текущего шага."""
    text = (state.get("input_text") or "").strip()
    step = state.get("step")

    # 0. Callback'и из inline-кнопок (list:prev, list:next и т.д.)
    if text == "list:prev":
        return "list_prev"
    if text == "list:next":
        return "list_next"
    if text.startswith("list:noop"):
        return "list_noop"
    if text == "list:edit":
        return "list_edit"
    if text == "list:edit:cancel":
        return "list_edit_cancel"
    if text.startswith("list:edit:"):
        return "list_edit_field"
    if text.startswith("list:set:"):
        return "list_edit_choice"
    if text == "list:delete":
        return "list_delete"
    if text == "list:delete_confirm":
        return "list_delete_do"
    if text == "list:delete_cancel":
        return "list_delete_cancel"
    if text == "list:toggle_photo":
        return "list_toggle_photo"
    if text == "list:delete_product_photo":
        return "list_delete_product_photo"
    if text == "list:make_product_photo":
        return "list_make_product_photo"

    # 0.1. Callback'и навигации по вариантам образа (/outfit)
    if text == "outfit:warmer":
        return "outfit_variant"
    if text == "outfit:lighter":
        return "outfit_variant"

    # 1. Slash-команды (работают всегда, даже посреди диалога)
    if text.startswith("/"):
        cmd = text.split()[0].lower()
        if cmd == "/start":
            return "start"
        if cmd in ("/help", "/info"):
            return "help"
        if cmd == "/cancel":
            return "cancel"
        if cmd == "/add":
            return "add_start"
        if cmd == "/list":
            return "list"
        if cmd == "/outfit":
            return "outfit_start"

    # 2. Кнопки главного меню (точное совпадение после удаления эмодзи)
    menu_intent = _match_menu_button(text)
    if menu_intent:
        return menu_intent

    # 3. Фото вне активного диалога — сразу начинаем добавление вещи
    if state.get("input_photo_file_id") and not step:
        return "add_photo"

    # 4. Не команда — смотрим на текущий шаг
    if step == "awaiting_photo":
        if state.get("input_photo_file_id"):
            return "add_photo"
        return "add_expect_photo"

    if step == "awaiting_confirm_category":
        return "add_confirm_category"

    if step == "awaiting_color":
        return "add_color"
    if step == "awaiting_color_text":
        return "add_color_text"

    if step == "awaiting_layer":
        return "add_layer"

    if step == "awaiting_location":
        return "outfit_location"

    if step == "awaiting_occasion":
        return "outfit_occasion"

    if step == "list_edit_input":
        return "list_edit_save"

    # На шаге выбора через кнопки — если пришёл текст (не callback),
    # обрабатываем как свободный ввод через list_edit_save.
    if step == "list_edit_choice" and not text.startswith("list:set:"):
        return "list_edit_save"

    if step == "awaiting_material":
        return "add_material"

    if step == "awaiting_material_text":
        return "add_material_text"

    if step and step.startswith("awaiting_"):
        return "add_attribute"

    return "fallback"


async def router_node(state: BotState) -> dict:
    return {}


def build_graph():
    builder = StateGraph(BotState)

    builder.add_node("router", router_node)
    builder.add_node("start", nodes.start_node)
    builder.add_node("help", nodes.help_node)
    builder.add_node("cancel", nodes.cancel_node)
    builder.add_node("fallback", nodes.fallback_node)
    builder.add_node("stub", nodes.stub_node)

    # /list и навигация по списку
    builder.add_node("list", nodes.list_node)
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

    # /outfit
    builder.add_node("outfit_start", nodes.outfit_start_node)
    builder.add_node("outfit_location", nodes.outfit_location_node)
    builder.add_node("outfit_occasion", nodes.outfit_occasion_node)
    builder.add_node("outfit_variant", nodes.outfit_variant_node)

    # /add
    builder.add_node("add_start", nodes.add_start_node)
    builder.add_node("add_expect_photo", nodes.add_expect_photo_node)
    builder.add_node("add_photo", nodes.add_photo_node)
    builder.add_node("add_confirm_category", nodes.add_confirm_category_node)
    builder.add_node("add_color", nodes.add_color_node)
    builder.add_node("add_color_text", nodes.add_color_text_node)
    builder.add_node("add_layer", nodes.add_layer_node)
    builder.add_node("add_attribute", nodes.add_attribute_node)
    builder.add_node("add_material", nodes.add_material_node)
    builder.add_node("add_material_text", nodes.add_material_text_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_condition,
        {
            "start": "start", "help": "help", "cancel": "cancel",
            "fallback": "fallback", "stub": "stub", "list": "list",
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
            "outfit_start": "outfit_start",
            "outfit_location": "outfit_location",
            "outfit_occasion": "outfit_occasion",
            "outfit_variant": "outfit_variant",
            "add_start": "add_start",
            "add_expect_photo": "add_expect_photo",
            "add_photo": "add_photo",
            "add_confirm_category": "add_confirm_category",
            "add_color": "add_color",
            "add_color_text": "add_color_text",
            "add_layer": "add_layer",
            "add_attribute": "add_attribute",
            "add_material": "add_material",
            "add_material_text": "add_material_text",
        },
    )

    terminal_nodes = [
        "start", "help", "cancel", "fallback", "stub", "list",
        "list_prev", "list_next", "list_noop", "list_edit", "list_delete",
        "list_delete_do", "list_delete_cancel",
        "list_toggle_photo", "list_make_product_photo",
        "list_delete_product_photo",
        "list_edit_field", "list_edit_choice", "list_edit_save", "list_edit_cancel",
        "outfit_start", "outfit_location", "outfit_occasion", "outfit_variant",
        "add_start", "add_expect_photo", "add_photo",
        "add_confirm_category", "add_attribute",
        "add_color", "add_color_text", "add_layer",
        "add_material", "add_material_text",
    ]
    for name in terminal_nodes:
        builder.add_edge(name, END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)