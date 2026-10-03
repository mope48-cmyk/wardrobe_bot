"""Сборка графа LangGraph."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import echo_node
from app.graph.state import BotState


def build_graph():
    """Создать и скомпилировать граф.

    Возвращает скомпилированный граф с подключённым checkpointer.
    """
    builder = StateGraph(BotState)

    # Добавляем узлы
    builder.add_node("echo", echo_node)

    # Определяем переходы
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)

    # MemorySaver хранит состояние в оперативной памяти.
    # При перезапуске бота история теряется — это нормально для этапа разработки.
    # На следующем этапе заменим на SqliteSaver, чтобы история сохранялась.
    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
