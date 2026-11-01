"""Тест синхронизации геймификации с фактической БД."""

from app.db.repository import add_item
from app.services.gamification import get_or_create_gam, level_from_xp


async def test_gam_syncs_items_from_db(db):
    """Если вещи уже были в БД, счётчик и XP подтягиваются."""
    uid = 424242
    for i in range(4):
        await add_item(
            user_id=uid,
            photo_file_id=f"ph{i}",
            category="верх",
            type=f"футболка{i}",
        )

    gam = await get_or_create_gam(uid)
    assert gam.items_added == 4
    # 4 вещи × 20 XP = 80 XP (уровень 1)
    assert gam.xp == 80
    assert level_from_xp(gam.xp) == 1


async def test_gam_sync_adds_missing_xp(db):
    """Вещи добавленные позже тоже досчитываются."""
    uid = 424243
    await add_item(user_id=uid, photo_file_id="ph0", category="верх", type="а")
    gam = await get_or_create_gam(uid)
    assert gam.items_added == 1

    # добавляем ещё 2 вещи
    await add_item(user_id=uid, photo_file_id="ph1", category="верх", type="б")
    await add_item(user_id=uid, photo_file_id="ph2", category="верх", type="в")
    gam = await get_or_create_gam(uid)
    assert gam.items_added == 3
    assert gam.xp == 60
