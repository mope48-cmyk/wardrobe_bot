"""Тесты геймификации (app/services/gamification.py)."""

import json
from datetime import date

from app.services.gamification import (
    CHALLENGES,
    level_from_xp,
    level_name,
    record_event,
    xp_for_level,
)

# ---------------------------------------------------------------------------
# Уровни
# ---------------------------------------------------------------------------


def test_xp_formula():
    assert xp_for_level(1) == 100
    assert xp_for_level(2) == 282
    assert xp_for_level(3) == 519
    assert level_from_xp(0) == 1
    assert level_from_xp(99) == 1
    assert level_from_xp(281) == 1
    assert level_from_xp(282) == 2
    assert level_name(1) == "Новичок"
    assert level_name(9) == "Легенда"


# ---------------------------------------------------------------------------
# record_event: XP и счётчики
# ---------------------------------------------------------------------------


async def test_record_event_adds_xp_and_counter(db):
    upd = await record_event(777, "item_added", today="2026-01-05")
    assert upd.xp_gained == 20
    assert upd.new_level == 1
    assert upd.leveled_up is False

    from app.services.gamification import get_or_create_gam
    gam = await get_or_create_gam(777)
    assert gam.items_added == 1
    assert gam.xp == 20


# ---------------------------------------------------------------------------
# Серия дней
# ---------------------------------------------------------------------------


async def test_streak_sequence(db):
    await record_event(777, "item_added", today="2026-01-05")
    await record_event(777, "item_added", today="2026-01-06")  # подряд
    from app.services.gamification import get_or_create_gam
    gam = await get_or_create_gam(777)
    assert gam.streak == 2

    # пропуск дня → сброс к 1
    await record_event(777, "item_added", today="2026-01-09")
    gam = await get_or_create_gam(777)
    assert gam.streak == 1

    # тот же день не увеличивает
    await record_event(777, "item_added", today="2026-01-09")
    gam = await get_or_create_gam(777)
    assert gam.streak == 1


async def test_streak_achievement(db):
    for day in range(1, 8):
        await record_event(777, "item_added", today=f"2026-01-{day:02d}")
    from app.services.gamification import get_or_create_gam
    gam = await get_or_create_gam(777)
    assert gam.streak >= 7
    assert "streak_7" in json.loads(gam.achievements)


# ---------------------------------------------------------------------------
# Достижения
# ---------------------------------------------------------------------------


async def test_first_item_achievement(db):
    upd = await record_event(777, "item_added", today="2026-01-05")
    assert "first_item" in upd.new_achievements

    from app.services.gamification import get_or_create_gam
    gam = await get_or_create_gam(777)
    assert "first_item" in json.loads(gam.achievements)


async def test_capsule_achievement(db):
    upd = await record_event(777, "capsule_built", today="2026-01-05")
    assert "capsule_first" in upd.new_achievements


# ---------------------------------------------------------------------------
# Недельный вызов
# ---------------------------------------------------------------------------


async def test_challenge_is_deterministic(db):
    await record_event(101, "item_added", today="2026-01-05")
    await record_event(102, "item_added", today="2026-01-05")

    from app.services.gamification import get_or_create_gam
    g1 = await get_or_create_gam(101)
    g2 = await get_or_create_gam(102)
    assert g1.weekly_challenge == g2.weekly_challenge
    assert g1.weekly_challenge in {c["type"] for c in CHALLENGES}


async def test_challenge_progress_and_completion(db):
    # Детерминированный вызов для недели 2026-01-05 (понедельник)
    week = date(2026, 1, 5).isocalendar()[1]
    challenge = CHALLENGES[(week - 1) % len(CHALLENGES)]
    expected_type = challenge["type"]
    target = challenge["target"]

    event_for_type = {
        "add_items": "item_added",
        "outfits": "outfit_created",
        "capsule": "capsule_built",
        "laundry": "laundry_done",
    }
    event = event_for_type[expected_type]

    upd = await record_event(777, event, today="2026-01-05")
    from app.services.gamification import get_or_create_gam
    gam = await get_or_create_gam(777)
    assert gam.weekly_challenge == expected_type, gam.weekly_challenge
    assert gam.challenge_progress == 1
    assert gam.challenge_target == target
    assert upd.challenge_completed is False

    # выполняем оставшиеся до цели; завершающее событие возвращает completed=True
    last_upd = upd
    for _ in range(target - 1):
        last_upd = await record_event(777, event, today="2026-01-05")
    gam = await get_or_create_gam(777)
    assert gam.challenge_progress == target
    assert last_upd.challenge_completed is True
