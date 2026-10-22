"""Состояние диалога для LangGraph."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class BotState(TypedDict, total=False):
    """Состояние диалога с пользователем. Все поля необязательные."""

    messages: Annotated[list, add_messages]

    user_id: int
    chat_id: int

    input_text: str | None
    input_photo_file_id: str | None
    input_photo_bytes: bytes | None
    input_location: dict | None

    intent: str | None
    step: str | None
    draft_item: dict | None

    location: dict | None
    occasion: str | None
    weather: dict | None
    outfit: dict | None

    # Для /list и любых списков
    list_ids: list[int] | None
    list_index: int
    list_source: str | None
    edit_message_id: int | None
    show_original: bool
    list_filter: str | None  # Текущий фильтр категории ("верх", "низ", "all" и т.д.)

    # Для /outfit: варианты подбора и текущий индекс
    outfit_options: list[list[int]] | None  # Список вариантов, каждый — список ID вещей
    outfit_index: int                       # Текущий индекс варианта
    outfit_city: str | None                 # Название города для комментария
    outfit_relaxed: list[list[str]] | None  # Флаги "плохого стиля" для каждого варианта

    # Для /add из альбома: очередь file_id на добавление
    album_queue: list[str] | None

    error: str | None