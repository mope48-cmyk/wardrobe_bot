"""Тесты веб-панели владельца (app/admin/web.py)."""

import sqlite3

import pytest

from app.admin import web
from app.admin.web import _level_from_xp, app

SCHEMA = """
CREATE TABLE wardrobe_items (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    photo_file_id TEXT NOT NULL,
    category TEXT NOT NULL,
    type TEXT NOT NULL,
    colors TEXT DEFAULT '[]',
    material TEXT DEFAULT '',
    warmth_level INTEGER DEFAULT 3,
    waterproof INTEGER DEFAULT 0,
    thickness INTEGER DEFAULT 3,
    has_hood INTEGER DEFAULT 0,
    use_white_overlay INTEGER DEFAULT 0,
    formal_level TEXT DEFAULT 'casual',
    season TEXT DEFAULT 'универсальная',
    layer TEXT DEFAULT 'base',
    accessory_occasions TEXT DEFAULT '[]',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_in_laundry INTEGER DEFAULT 0,
    laundry_start_date DATETIME
);
CREATE TABLE user_morning_settings (
    user_id INTEGER PRIMARY KEY,
    enabled INTEGER DEFAULT 0,
    time_str TEXT DEFAULT '08:00',
    city TEXT,
    lat REAL,
    lon REAL,
    occasion TEXT DEFAULT 'прогулка',
    days TEXT DEFAULT '0,1,2,3,4,5,6',
    last_sent_date TEXT
);
CREATE TABLE user_gamification (
    user_id INTEGER PRIMARY KEY,
    xp INTEGER DEFAULT 0,
    streak INTEGER DEFAULT 0,
    last_active_date TEXT,
    weekly_challenge TEXT,
    challenge_progress INTEGER DEFAULT 0,
    challenge_target INTEGER DEFAULT 0,
    challenge_week INTEGER DEFAULT 0,
    achievements TEXT DEFAULT '[]',
    items_added INTEGER DEFAULT 0,
    outfits_created INTEGER DEFAULT 0,
    verdicts_done INTEGER DEFAULT 0,
    capsules_built INTEGER DEFAULT 0,
    laundry_done INTEGER DEFAULT 0
);
CREATE TABLE user_preferences (
    user_id INTEGER PRIMARY KEY,
    disliked_items TEXT DEFAULT '[]'
);
"""


@pytest.fixture
def panel_db(tmp_path, monkeypatch):
    path = tmp_path / "admin_test.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executescript(
        """
        INSERT INTO user_gamification
            (user_id, xp, streak, last_active_date, achievements,
             items_added, outfits_created, verdicts_done,
             capsules_built, laundry_done)
        VALUES
            (111, 320, 4, '2026-10-31', '["first_item"]',
             9, 3, 1, 0, 2),
            (222, 55, 1, '2026-11-01', '[]',
             2, 0, 0, 0, 0);
        INSERT INTO wardrobe_items (id, user_id, photo_file_id, category, type)
        VALUES
            (1, 111, 'f1', 'верх', 'футболка'),
            (2, 111, 'f2', 'низ', 'джинсы'),
            (3, 222, 'f3', 'обувь', 'кроссовки');
        INSERT INTO user_morning_settings (user_id, enabled, time_str, city)
        VALUES (111, 1, '08:30', 'Москва');
        INSERT INTO user_preferences (user_id, disliked_items)
        VALUES (111, '[2]');
        """
    )
    con.commit()
    con.close()

    monkeypatch.setattr(web, "_db_path", lambda: str(path))
    monkeypatch.setenv("ADMIN_TOKEN", "secret-token")
    return path


def test_db_path_parsing(monkeypatch):
    cases = {
        "sqlite+aiosqlite:////app/data/wardrobe.db": "/app/data/wardrobe.db",
        "sqlite+aiosqlite:///./ci_test.db": "/./ci_test.db",
        "data/db.sqlite": "data/db.sqlite",
    }
    for url, expected in cases.items():
        monkeypatch.setenv("DATABASE_URL", url)
        assert web._db_path() == expected

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("WARDROBE_DB", "/custom/db.sqlite")
    assert web._db_path() == "/custom/db.sqlite"


def test_level_from_xp():
    # Пороги: уровень L требует int(100 * L**1.5) XP
    assert _level_from_xp(0) == 1
    assert _level_from_xp(281) == 1
    assert _level_from_xp(282) == 2
    assert _level_from_xp(518) == 2
    assert _level_from_xp(519) == 3
    assert _level_from_xp(799) == 3
    assert _level_from_xp(800) == 4
    assert _level_from_xp(1117) == 4
    assert _level_from_xp(1118) == 5


def test_panel_requires_auth(panel_db):
    client = app.test_client()
    assert client.get("/").status_code == 401
    assert client.get("/?key=wrong").status_code == 403
    assert client.get("/?key=secret-token").status_code == 200


def test_panel_shows_stats_and_users(panel_db):
    client = app.test_client()
    resp = client.get("/?key=secret-token")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert "Панель владельца" in html
    assert "111" in html and "222" in html
    assert "Москва" in html
    assert "вкл 08:30" in html
    assert "Дизлайки" in html
    # Карточки агрегатов: 2 пользователя, 3 вещи, 375 XP суммарно.
    assert "<b>2</b>" in html
    assert "<b>3</b>" in html
    assert "<b>375</b>" in html


def test_panel_sorting_and_cookie(panel_db):
    client = app.test_client()
    assert client.get("/?key=secret-token&sort=xp&order=desc").status_code == 200
    assert client.get("/?sort=user_id").status_code == 200  # cookie уже стоит
