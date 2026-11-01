"""Тесты узла «ИИ-Анализ» вещи (list_ai_analyze_node)."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.graph.nodes import list_ai_analyze_node


@pytest.mark.asyncio
async def test_ai_analyze_writes_all_fields_including_thickness_hood(monkeypatch):
    """GigaChat-анализ пишет в БД все 11 полей, включая thickness/has_hood."""
    captured = []

    fake_item = SimpleNamespace(
        id=7, photo_file_id="f_photo", category="верх", type="куртка",
        colors="[]", material="", warmth_level=3, waterproof=False,
        formal_level="casual", season="универсальная", layer="outer",
        thickness=3, has_hood=False,
    )

    async def fake_get_item(item_id):
        return fake_item

    async def fake_download(file_id):
        return b"jpeg-bytes"

    async def fake_analyze(image_bytes):
        return {
            "category": "верх",
            "type": "куртка",
            "colors": ["чёрный"],
            "material": "кожа",
            "warmth_level": 4,
            "waterproof": True,
            "formal_level": "casual",
            "season": "зима",
            "layer": "outer",
            "thickness": 5,
            "has_hood": True,
        }

    async def fake_update(item_id, user_id, field, value):
        captured.append((field, value))
        return True

    async def fake_render(state, edit_mode=None):
        return {"messages": [AIMessage(content="карточка")]}

    monkeypatch.setattr("app.graph.nodes.get_item", fake_get_item)
    monkeypatch.setattr(
        "app.services.telegram_files.download_photo_by_file_id", fake_download
    )
    monkeypatch.setattr(
        "app.services.vision_gigachat.analyze_clothing_with_gigachat",
        fake_analyze,
    )
    monkeypatch.setattr("app.graph.nodes.update_item_field", fake_update)
    monkeypatch.setattr("app.graph.nodes._render_current", fake_render)

    state = {"user_id": 1, "list_ids": [7], "list_index": 0}
    result = await list_ai_analyze_node(state)

    assert result["messages"], "должна вернуться перерисованная карточка"

    fields = {field for field, _ in captured}
    assert "thickness" in fields
    assert "has_hood" in fields
    assert ("thickness", 5) in captured
    assert ("has_hood", True) in captured
    # остальные ключевые поля тоже записываются
    for field in ("category", "type", "colors", "material", "warmth_level",
                  "waterproof", "formal_level", "season", "layer"):
        assert field in fields
