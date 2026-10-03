"""Состояние диалога для LangGraph."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class BotState(TypedDict, total=False):
    """Состояние диалога с пользователем.

    total=False — все поля необязательные. Это упрощает работу:
    не нужно каждый раз указывать все поля в input_state.
    """

    # История сообщений. add_messages добавляет новые сообщения
    # к существующему списку (не перезаписывает).
    messages: Annotated[list, add_messages]

    # Идентификация
    user_id: int
    chat_id: int

    # Разобранный ввод пользователя (обновляется на каждом сообщении).
    # Хранит структурированное представление последнего входящего сообщения.
    input_text: str | None
    input_photo_file_id: str | None
    input_location: dict | None

    # Состояние диалога
    intent: str | None
    step: str | None
    draft_item: dict | None

    # Для будущих сценариев (/outfit)
    location: dict | None
    occasion: str | None
    weather: dict | None
    outfit: dict | None

    # Ошибки
    error: str | None