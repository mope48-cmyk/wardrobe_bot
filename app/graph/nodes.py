"""Узлы графа LangGraph."""

import logging

from langchain_core.messages import AIMessage

from app.graph.state import BotState

logger = logging.getLogger(__name__)


async def echo_node(state: BotState) -> dict:
    """Временный узел-эхо.

    Берёт последнее сообщение пользователя и возвращает его же
    с префиксом. Используется, чтобы проверить, что граф и память
    работают корректно.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else ""

    logger.info("echo_node: user_id=%s text=%r", state["user_id"], user_text)

    # Возвращаем словарь — LangGraph сам смёржит его с текущим состоянием.
    # Поле messages объединится с существующим списком благодаря add_messages.
    return {
        "messages": [
            AIMessage(content=f"Эхо: {user_text}")
        ]
    }
