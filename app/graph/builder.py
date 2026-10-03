"""Сборка графа LangGraph."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import BotState


def route_condition(state: BotState) -> str:
    text = (state.get("input_text") or "").strip()
    first_word = text.split()[0].lower() if text else ""

    if first_word == "/start": return "start"
    if first_word == "/help": return "help"
    if first_word == "/cancel": return "cancel"
    if first_word == "/add": return "add_start"
    if first_word == "/list": return "list"
    if first_word == "/outfit": return "stub"

    step = state.get("step")

    if step == "awaiting_photo":
        if state.get("input_photo_file_id"):
            return "add_photo"
        return "add_expect_photo"

    if step == "awaiting_confirm_category":
        return "add_confirm_category"

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
    builder.add_node("list", nodes.list_node)
    builder.add_node("add_start", nodes.add_start_node)
    builder.add_node("add_expect_photo", nodes.add_expect_photo_node)
    builder.add_node("add_photo", nodes.add_photo_node)
    builder.add_node("add_confirm_category", nodes.add_confirm_category_node)
    builder.add_node("add_attribute", nodes.add_attribute_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_condition,
        {
            "start": "start", "help": "help", "cancel": "cancel",
            "fallback": "fallback", "stub": "stub", "list": "list",
            "add_start": "add_start",
            "add_expect_photo": "add_expect_photo",
            "add_photo": "add_photo",
            "add_confirm_category": "add_confirm_category",
            "add_attribute": "add_attribute",
        },
    )

    terminal_nodes = [
        "start", "help", "cancel", "fallback", "stub", "list",
        "add_start", "add_expect_photo", "add_photo",
        "add_confirm_category", "add_attribute",
    ]
    for name in terminal_nodes:
        builder.add_edge(name, END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)