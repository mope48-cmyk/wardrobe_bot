"""Тесты обратной связи на образы (app/services/feedback.py)."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, OutfitFeedback


@pytest.fixture
async def feedback_db(tmp_path):
    """Изолированная БД для фидбек-а + подменённый SessionLocal."""
    path = tmp_path / "fb_test.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{path.as_posix()}"
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)

    from app.services import feedback as fbmod
    fbmod.SessionLocal = SessionLocal
    yield SessionLocal
    await engine.dispose()


@pytest.mark.asyncio
async def test_record_outfit_feedback_like(feedback_db):
    SessionLocal = feedback_db
    from app.services.feedback import record_outfit_feedback

    await record_outfit_feedback(
        user_id=111,
        verdict="like",
        occasion="прогулка",
        weather_main="ясно",
        weather_temp=15.0,
        outfit_types=["футболка", "джинсы"],
    )

    async with SessionLocal() as session:
        total = await session.scalar(
            select(func.count()).select_from(OutfitFeedback)
        )
        row = await session.scalar(select(OutfitFeedback))
        assert total == 1
        assert row.user_id == 111
        assert row.verdict == "like"
        assert row.occasion == "прогулка"
        assert row.weather_main == "ясно"
        assert row.weather_temp == 15.0
        assert row.outfit_types == '["футболка", "джинсы"]'


@pytest.mark.asyncio
async def test_record_outfit_feedback_invalid_verdict(feedback_db):
    SessionLocal = feedback_db
    from app.services.feedback import record_outfit_feedback

    await record_outfit_feedback(user_id=111, verdict="meh")

    async with SessionLocal() as session:
        total = await session.scalar(
            select(func.count()).select_from(OutfitFeedback)
        )
        assert total == 0


@pytest.mark.asyncio
async def test_outfit_feedback_node_like(monkeypatch):
    """Узел сохраняет оценку и отвечает вежливо."""
    from types import SimpleNamespace

    from app.graph.nodes import outfit_feedback_node

    captured = {}

    async def fake_record(**kwargs):
        captured.update(kwargs)

    async def fake_get_items(user_id):
        item = SimpleNamespace(
            id=1, type="футболка", category="верх",
            photo_file_id="f1", generated_photo_file_id=None,
            colors="[]", material="", warmth_level=3, waterproof=False,
            thickness=3, has_hood=False, use_white_overlay=False,
            formal_level="casual", season="универсальная", layer="base",
            accessory_occasions="[]", is_in_laundry=False,
            laundry_start_date=None,
        )
        return [item]

    monkeypatch.setattr("app.services.feedback.record_outfit_feedback", fake_record)
    monkeypatch.setattr("app.graph.nodes.get_items", fake_get_items)

    state = {
        "user_id": 111,
        "input_text": "outfit:fb:like",
        "occasion": "прогулка",
        "weather": {"description": "ясно", "temp": 15.0},
        "outfit_options": [[1]],
        "outfit_index": 0,
    }

    result = await outfit_feedback_node(state)

    assert captured["user_id"] == 111
    assert captured["verdict"] == "like"
    assert captured["occasion"] == "прогулка"
    assert captured["weather_main"] == "ясно"
    assert captured["weather_temp"] == 15.0
    assert captured["outfit_types"] == ["футболка"]

    assert result["step"] is None
    assert len(result["messages"]) == 1
    assert "Спасибо" in result["messages"][0].content

