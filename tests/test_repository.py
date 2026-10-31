"""Тесты репозитория на временной SQLite-БД (app/db/repository.py)."""


from app.db.repository import (
    add_disliked_item,
    add_item,
    delete_item,
    get_disliked_items,
    get_item,
    get_items,
    get_morning_settings,
    get_users_for_morning_push,
    update_item_field,
    update_morning_settings,
)

# ---------------------------------------------------------------------------
# WardrobeItem CRUD
# ---------------------------------------------------------------------------


async def test_add_and_get_item(db):
    item = await add_item(
        user_id=42,
        photo_file_id="ph1",
        category="верх",
        type="футболка",
        colors=["белый"],
        material="хлопок",
        warmth_level=2,
        waterproof=False,
        thickness=1,
        has_hood=False,
    )
    assert item.id is not None
    assert item.use_white_overlay is False

    fetched = await get_item(item.id)
    assert fetched is not None
    assert fetched.type == "футболка"
    assert fetched.thickness == 1

    all_items = await get_items(42)
    assert any(i.id == item.id for i in all_items)


async def test_update_item_field_allowed_and_forbidden(db):
    item = await add_item(user_id=42, photo_file_id="ph1", category="верх", type="рубашка")

    assert await update_item_field(item.id, 42, "thickness", 4) is True
    assert await update_item_field(item.id, 42, "has_hood", True) is True
    assert await update_item_field(item.id, 42, "use_white_overlay", True) is True

    # Запрещённое поле — False, без изменения
    assert await update_item_field(item.id, 42, "user_id", 999) is False
    # Чужой user_id — False
    assert await update_item_field(item.id, 999, "thickness", 1) is False

    fetched = await get_item(item.id)
    assert fetched.thickness == 4
    assert fetched.has_hood is True
    assert fetched.use_white_overlay is True
    assert fetched.user_id == 42


async def test_delete_item(db):
    item = await add_item(user_id=42, photo_file_id="ph1", category="верх", type="рубашка")
    assert await delete_item(item.id, 42) is True
    assert await get_item(item.id) is None
    # Повторное удаление — False
    assert await delete_item(item.id, 42) is False


# ---------------------------------------------------------------------------
# Предпочтения (дизлайки)
# ---------------------------------------------------------------------------


async def test_disliked_preferences_roundtrip(db):
    assert await get_disliked_items(42) == []

    await add_disliked_item(42, 1)
    await add_disliked_item(42, 2)
    await add_disliked_item(42, 1)  # дубликат не добавится

    assert await get_disliked_items(42) == [1, 2]


async def test_disliked_preferences_autocreate(db):
    await add_disliked_item(7, 100)
    assert await get_disliked_items(7) == [100]


# ---------------------------------------------------------------------------
# Утренняя сводка
# ---------------------------------------------------------------------------


async def test_morning_settings_upsert(db):
    settings = await get_morning_settings(42)
    assert settings.enabled is False
    assert settings.time_str == "08:00"

    await update_morning_settings(
        42, enabled=True, time_str="07:30", days="1,2,3", city="Москва"
    )
    updated = await get_morning_settings(42)
    assert updated.enabled is True
    assert updated.time_str == "07:30"
    assert updated.days == "1,2,3"


async def test_get_users_for_morning_push_filters(db):
    # Сначала создаём строки (update_morning_settings — UPDATE без upsert)
    for uid in (42, 43, 44):
        await get_morning_settings(uid)
    await update_morning_settings(42, enabled=True, time_str="08:00", days="1,2,3,4,5,6,0")
    await update_morning_settings(43, enabled=False, time_str="08:00", days="1")
    await update_morning_settings(44, enabled=True, time_str="09:00", days="1")

    users = await get_users_for_morning_push("08:00", 1)  # Пн
    user_ids = {u.user_id for u in users}
    assert 42 in user_ids      # enabled + время + день подходят
    assert 43 not in user_ids  # disabled
    assert 44 not in user_ids  # не то время
