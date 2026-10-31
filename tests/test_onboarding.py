"""Тесты онбординга (app/graph/nodes.py)."""

import pytest

from app.graph import nodes as N


async def _fake_get_items(items):
    async def _inner(user_id):
        return items
    return _inner


# ---------------------------------------------------------------------------
# _get_onboarding_state
# ---------------------------------------------------------------------------


async def test_onboarding_empty_suggests_top(monkeypatch, make_item):
    monkeypatch.setattr(N, "get_items", await _fake_get_items([]))
    text, ready = await N._get_onboarding_state(1)
    assert ready is False
    assert "(0/5)" in text
    assert "футболку или рубашку" in text


async def test_onboarding_progress_bar(monkeypatch, make_item):
    items = [make_item() for _ in range(3)]
    monkeypatch.setattr(N, "get_items", await _fake_get_items(items))
    text, ready = await N._get_onboarding_state(1)
    assert ready is False
    assert "🟩🟩🟩" in text and "(3/5)" in text


async def test_onboarding_suggestions_sequence(monkeypatch, make_item):
    cases = [
        ([], "футболку или рубашку"),
        ([make_item(category="верх")], "джинсы или брюки"),
        ([make_item(category="верх"), make_item(category="низ")], "кроссовки или ботинки"),
        (
            [make_item(category="верх"), make_item(category="низ"), make_item(category="обувь")],
            "куртку или пальто",
        ),
        (
            [
                make_item(category="верх"),
                make_item(category="низ"),
                make_item(category="обувь"),
                make_item(category="верх", type="куртка", layer="outer"),
            ],
            "ещё одну вещь",
        ),
    ]
    for items, expected_suggestion in cases:
        monkeypatch.setattr(N, "get_items", await _fake_get_items(items))
        text, ready = await N._get_onboarding_state(1)
        assert ready is False
        assert expected_suggestion in text, text


async def test_onboarding_ready_at_five(monkeypatch, make_item):
    items = [make_item() for _ in range(5)]
    monkeypatch.setattr(N, "get_items", await _fake_get_items(items))
    text, ready = await N._get_onboarding_state(1)
    assert ready is True
    assert text == N.WELCOME_TEXT


# ---------------------------------------------------------------------------
# _persist_item: ветки
# ---------------------------------------------------------------------------


DRAFT = {
    "photo_file_id": "photo1",
    "category": "верх",
    "type": "футболка",
    "colors": ["белый"],
    "material": "хлопок",
    "warmth_level": 2,
    "waterproof": False,
    "formal_level": "casual",
    "season": "лето",
    "layer": "base",
}


async def test_persist_celebration_at_exactly_5(monkeypatch, make_item):
    captured = {}

    async def fake_add_item(**kwargs):
        captured["kwargs"] = kwargs
        return None

    async def fake_get_items(user_id):
        return [make_item(id=i) for i in range(5)]

    async def fake_onboarding(user_id):
        return "TEXT", True

    monkeypatch.setattr(N, "add_item", fake_add_item)
    monkeypatch.setattr(N, "get_items", fake_get_items)
    monkeypatch.setattr(N, "_get_onboarding_state", fake_onboarding)

    result = await N._persist_item({"user_id": 1, "album_queue": None}, DRAFT)
    content = result["messages"][0].content
    assert "У вас 5 вещей!" in content
    assert captured["kwargs"]["thickness"] == 3
    assert captured["kwargs"]["has_hood"] is False


async def test_persist_ordinary_after_5(monkeypatch, make_item):
    async def fake_add_item(**kwargs):
        return None

    async def fake_get_items(user_id):
        return [make_item(id=i) for i in range(6)]

    async def fake_onboarding(user_id):
        return "TEXT", True

    monkeypatch.setattr(N, "add_item", fake_add_item)
    monkeypatch.setattr(N, "get_items", fake_get_items)
    monkeypatch.setattr(N, "_get_onboarding_state", fake_onboarding)

    result = await N._persist_item({"user_id": 1, "album_queue": None}, DRAFT)
    assert "Что-нибудь ещё?" in result["messages"][0].content


async def test_persist_under_5_appends_onboarding(monkeypatch, make_item):
    async def fake_add_item(**kwargs):
        return None

    async def fake_get_items(user_id):
        return [make_item(id=i) for i in range(3)]

    async def fake_onboarding(user_id):
        return "ОНБОРДИНГ-ТЕКСТ", False

    monkeypatch.setattr(N, "add_item", fake_add_item)
    monkeypatch.setattr(N, "get_items", fake_get_items)
    monkeypatch.setattr(N, "_get_onboarding_state", fake_onboarding)

    result = await N._persist_item({"user_id": 1, "album_queue": None}, DRAFT)
    assert "ОНБОРДИНГ-ТЕКСТ" in result["messages"][0].content
