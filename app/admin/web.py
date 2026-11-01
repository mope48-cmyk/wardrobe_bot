"""Веб-панель владельца бота (чтение SQLite БД, без записи).

Отдельный сервис в docker-compose, слушает 127.0.0.1:8080.
Доступ защищён токеном из переменной окружения ADMIN_TOKEN.
"""

import functools
import hmac
import json
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, make_response, render_template, request

MSC = ZoneInfo("Europe/Moscow")

LEVEL_NAMES = {
    1: "Новичок",
    2: "Начинающий модник",
    3: "Обыватель стиля",
    4: "Стилист",
    5: "Гуру моды",
    6: "Икона",
}

SORT_WHITELIST = {
    "user_id": "user_id",
    "items": "item_count",
    "outfits": "outfits",
    "verdicts": "verdicts",
    "capsules": "capsules",
    "laundry": "laundry",
    "xp": "xp",
    "streak": "streak",
    "last_active": "last_active",
}

app = Flask(__name__)


def _check_token(token: str | None) -> bool:
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    if not expected or not token:
        return False
    return hmac.compare_digest(expected, token)


def require_auth(view):
    """Требует корректный ADMIN_TOKEN (?key= или cookie); ставит cookie."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        token = request.args.get("key") or request.cookies.get("admin_key")
        if not _check_token(token):
            if request.args.get("key"):
                return "Неверный ключ доступа", 403
            return render_template("login.html"), 401
        resp = make_response(view(*args, **kwargs))
        if not request.cookies.get("admin_key"):
            resp.set_cookie("admin_key", token, httponly=True, samesite="Lax")
        return resp

    return wrapped


def _db_path() -> str:
    """Извлекает путь к файлу SQLite из DATABASE_URL или WARDROBE_DB."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        rest = url.split("://", 1)[1] if "://" in url else url
        if rest.startswith("//"):
            return "/" + rest[2:]
        return rest
    return os.getenv("WARDROBE_DB", "/app/data/wardrobe.db")


def _level_from_xp(xp: int) -> int:
    level = 1
    while xp >= int(100 * (level + 1) ** 1.5):
        level += 1
    return level


def _json_count(raw: str | None) -> int:
    if not raw or not raw.strip():
        return 0
    try:
        return len(json.loads(raw or "[]"))
    except Exception:
        return 0


def _fetch_rows() -> list[dict]:
    """Все пользователи с агрегированной статистикой по фичам."""
    con = sqlite3.connect(_db_path())
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()

        cur.execute(
            """
            SELECT g.user_id AS user_id,
                   g.xp AS xp,
                   g.streak AS streak,
                   g.last_active_date AS last_active,
                   g.items_added AS items,
                   g.outfits_created AS outfits,
                   g.verdicts_done AS verdicts,
                   g.capsules_built AS capsules,
                   g.laundry_done AS laundry,
                   g.achievements AS achievements,
                   m.enabled AS morning_enabled,
                   m.time_str AS morning_time,
                   m.city AS morning_city,
                   m.days AS morning_days
            FROM user_gamification g
            LEFT JOIN user_morning_settings m ON m.user_id = g.user_id
            ORDER BY g.user_id
            """
        )
        rows = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT user_id, COUNT(*) AS cnt, MAX(created_at) AS last_item "
            "FROM wardrobe_items GROUP BY user_id"
        )
        item_stats = {r["user_id"]: dict(r) for r in cur.fetchall()}

        cur.execute(
            "SELECT user_id, COUNT(*) AS cnt FROM wardrobe_items "
            "WHERE is_in_laundry = 1 GROUP BY user_id"
        )
        laundry_now = {r["user_id"]: r["cnt"] for r in cur.fetchall()}

        cur.execute("SELECT user_id, disliked_items FROM user_preferences")
        dislikes = {r["user_id"]: r["disliked_items"] for r in cur.fetchall()}

        today = datetime.now(MSC).strftime("%Y-%m-%d")
        for row in rows:
            uid = row["user_id"]
            stats = item_stats.get(uid, {})
            row["item_count"] = stats.get("cnt", 0)
            row["last_item"] = stats.get("last_item")
            row["laundry_now"] = laundry_now.get(uid, 0)
            row["disliked"] = _json_count(dislikes.get(uid))
            row["active_today"] = row["last_active"] == today
            row["level"] = _level_from_xp(row["xp"] or 0)
            row["level_name"] = LEVEL_NAMES.get(row["level"], "Легенда")
            row["ach_count"] = _json_count(row["achievements"])
        return rows
    finally:
        con.close()


def _aggregate(rows: list[dict]) -> dict:
    return {
        "users": len(rows),
        "items": sum(r["item_count"] for r in rows),
        "active_today": sum(1 for r in rows if r["active_today"]),
        "morning_enabled": sum(1 for r in rows if r["morning_enabled"]),
        "outfits": sum(r["outfits"] or 0 for r in rows),
        "verdicts": sum(r["verdicts"] or 0 for r in rows),
        "capsules": sum(r["capsules"] or 0 for r in rows),
        "laundry": sum(r["laundry"] or 0 for r in rows),
        "xp": sum(r["xp"] or 0 for r in rows),
    }


def _feedback_stats() -> dict:
    """Оценки образов (кнопки 👍/👎): всего, лайки, дизлайки, рейтинг."""
    try:
        con = sqlite3.connect(_db_path())
        con.row_factory = sqlite3.Row
        try:
            likes = con.execute(
                "SELECT COUNT(*) FROM outfit_feedback WHERE verdict='like'"
            ).fetchone()[0]
            dislikes = con.execute(
                "SELECT COUNT(*) FROM outfit_feedback WHERE verdict='dislike'"
            ).fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return {"total": 0, "likes": 0, "dislikes": 0, "match_rate": 0.0}

    total = likes + dislikes
    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "match_rate": round(likes / total, 3) if total else 0.0,
    }


def _sort_key(row: dict, key: str):
    value = row[key]
    return (value is None, value)


@app.get("/")
@require_auth
def index():
    sort_by = request.args.get("sort", "user_id")
    order = request.args.get("order", "asc")
    if sort_by not in SORT_WHITELIST:
        sort_by = "user_id"
    key = SORT_WHITELIST[sort_by]
    reverse = order == "desc"

    try:
        rows = _fetch_rows()
    except sqlite3.Error as exc:
        return render_template("admin.html", error=f"Ошибка чтения БД: {exc}"), 500

    rows.sort(key=lambda r: _sort_key(r, key), reverse=reverse)
    agg = _aggregate(rows)
    agg["feedback"] = _feedback_stats()
    return render_template(
        "admin.html",
        rows=rows,
        agg=agg,
        sort=sort_by,
        order=order,
    )


if __name__ == "__main__":
    # Внутри контейнера слушаем 0.0.0.0, чтобы docker-proxy мог достучаться.
    # Наружу порт НЕ торчит: в docker-compose проброс ограничен 127.0.0.1.
    app.run(host="0.0.0.0", port=int(os.getenv("ADMIN_PORT", "8080")))
