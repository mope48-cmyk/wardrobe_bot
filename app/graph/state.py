"""Состояние диалога для LangGraph."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class BotState(TypedDict):
    """Состояние диалога с пользователем.

    Поля, которые заполняются постепенно, могут быть None,
    пока соответствующий шаг диалога не начался.
    """

    # История сообщений. add_messages — специальный редьюсер LangGraph,
    # который добавляет новые сообщения к существующему списку
    # (а не перезаписывает его).
    messages: Annotated[list, add_messages]

    # Идентификация
    user_id: int
    chat_id: int

    # Текущее намерение и шаг диалога
    intent: str | None      # add / list / outfit / help / None
    step: str | None        # awaiting_photo, awaiting_category и т.д.

    # Черновик вещи, собираемой в /add
    draft_item: dict | None

    # Данные для /outfit
    location: dict | None   # {"lat": ..., "lon": ..., "city": ...}
    occasion: str | None
    weather: dict | None
    outfit: dict | None

    # Ошибки (для отладки)
    error: str | None
