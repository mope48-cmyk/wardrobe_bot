"""Геймификация: XP/уровни, серия дней, достижения, недельный вызов.

Все события записываются через единую точку record_event(), которая:
  1. обновляет серию дней (streak);
  2. начисляет XP и увеличивает счётчик события;
  3. продвигает недельный вызов;
  4. проверяет и разблокирует достижения.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from app.db.database import SessionLocal
from app.db.models import UserGamification

logger = logging.getLogger(__name__)

MSC = ZoneInfo("Europe/Moscow")
# ---------------------------------------------------------------------------
# Уровни (XP = 100 * level^1.5, уровень стартует с 1)
# ---------------------------------------------------------------------------

LEVEL_NAMES = {
    1: "Новичок",
    2: "Начинающий модник",
    3: "Обыватель стиля",
    4: "Стилист",
    5: "Гуру моды",
    6: "Икона",
}


def xp_for_level(level: int) -> int:
    """Сколько XP нужно ДЛЯ достижения уровня level."""
    return int(100 * level ** 1.5)


def level_from_xp(xp: int) -> int:
    level = 1
    while xp >= xp_for_level(level + 1):
        level += 1
    return level


def level_name(level: int) -> str:
    return LEVEL_NAMES.get(level, "Легенда")


# ---------------------------------------------------------------------------
# События: XP и счётчики
# ---------------------------------------------------------------------------

EVENT_XP = {
    "item_added": 20,
    "outfit_created": 15,
    "verdict_done": 25,
    "capsule_built": 40,
    "laundry_done": 10,
}

EVENT_COUNTER = {
    "item_added": "items_added",
    "outfit_created": "outfits_created",
    "verdict_done": "verdicts_done",
    "capsule_built": "capsules_built",
    "laundry_done": "laundry_done",
}

# ---------------------------------------------------------------------------
# Недельный вызов
# ---------------------------------------------------------------------------

CHALLENGES = [
    {"type": "add_items", "target": 3, "title": "Добавь 3 вещи"},
    {"type": "outfits", "target": 5, "title": "Собери 5 образов"},
    {"type": "capsule", "target": 1, "title": "Создай капсулу"},
    {"type": "laundry", "target": 2, "title": "Отправь 2 вещи в стирку"},
]

EVENT_TO_CHALLENGE = {
    "item_added": "add_items",
    "outfit_created": "outfits",
    "capsule_built": "capsule",
    "laundry_done": "laundry",
}

CHALLENGE_BONUS_XP = 50

# ---------------------------------------------------------------------------
# Достижения
# ---------------------------------------------------------------------------

ACHIEVEMENTS = {
    "first_item": {
        "emoji": "👟", "title": "Первый шаг",
        "desc": "Добавьте первую вещь",
        "check": lambda g: g.items_added >= 1,
    },
    "wardrobe_10": {
        "emoji": "🧥", "title": "Гардероб растёт",
        "desc": "10 вещей", "check": lambda g: g.items_added >= 10,
    },
    "wardrobe_25": {
        "emoji": "🗄️", "title": "Серьёзная коллекция",
        "desc": "25 вещей", "check": lambda g: g.items_added >= 25,
    },
    "wardrobe_50": {
        "emoji": "🏛️", "title": "Половина бутика",
        "desc": "50 вещей", "check": lambda g: g.items_added >= 50,
    },
    "wardrobe_100": {
        "emoji": "👑", "title": "Император гардероба",
        "desc": "100 вещей", "check": lambda g: g.items_added >= 100,
    },
    "stylist_5": {
        "emoji": "✨", "title": "Настоящий стилист",
        "desc": "5 собранных образов", "check": lambda g: g.outfits_created >= 5,
    },
    "verdict_first": {
        "emoji": "⚖️", "title": "Смельчак",
        "desc": "Первый приговор", "check": lambda g: g.verdicts_done >= 1,
    },
    "verdict_5": {
        "emoji": "🗡️", "title": "Беспощадный критик",
        "desc": "5 приговоров", "check": lambda g: g.verdicts_done >= 5,
    },
    "capsule_first": {
        "emoji": "🎒", "title": "Капсулятор",
        "desc": "Первая капсула", "check": lambda g: g.capsules_built >= 1,
    },
    "laundry_5": {
        "emoji": "🧺", "title": "Чистюля",
        "desc": "5 отправок в стирку", "check": lambda g: g.laundry_done >= 5,
    },
    "streak_7": {
        "emoji": "🔥", "title": "Неделя в ритме моды",
        "desc": "Серия 7 дней", "check": lambda g: g.streak >= 7,
    },
    "streak_30": {
        "emoji": "💫", "title": "Месяц стиля",
        "desc": "Серия 30 дней", "check": lambda g: g.streak >= 30,
    },
    "level_3": {
        "emoji": "🎩", "title": "Растущий талант",
        "desc": "Уровень 3", "check": lambda g: level_from_xp(g.xp) >= 3,
    },
    "level_5": {
        "emoji": "🏆", "title": "Маэстро стиля",
        "desc": "Уровень 5", "check": lambda g: level_from_xp(g.xp) >= 5,
    },
}

# ---------------------------------------------------------------------------
# Dataclass результата
# ---------------------------------------------------------------------------


@dataclass
class GamUpdate:
    new_achievements: list[str] = field(default_factory=list)
    leveled_up: bool = False
    new_level: int = 0
    challenge_completed: bool = False
    xp_gained: int = 0


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def _yesterday_str(today: str) -> str:
    return (date.fromisoformat(today) - timedelta(days=1)).strftime("%Y-%m-%d")


def _iso_week(today: str) -> int:
    return date.fromisoformat(today).isocalendar()[1]


def _update_streak(gam: UserGamification, today: str) -> None:
    if gam.last_active_date == today:
        return
    if gam.last_active_date:
        gam.streak = (
            gam.streak + 1
            if gam.last_active_date == _yesterday_str(today)
            else 1
        )
    else:
        gam.streak = 1
    gam.last_active_date = today


def _update_challenge(gam: UserGamification, event: str, today: str) -> bool:
    """Обновить недельный вызов. Возвращает True, если вызов только что выполнен."""
    week = _iso_week(today)
    if not gam.weekly_challenge or gam.challenge_week != week:
        ch = CHALLENGES[(week - 1) % len(CHALLENGES)]
        gam.weekly_challenge = ch["type"]
        gam.challenge_target = ch["target"]
        gam.challenge_progress = 0
        gam.challenge_week = week

    if EVENT_TO_CHALLENGE.get(event) != gam.weekly_challenge:
        return False
    if gam.challenge_progress >= gam.challenge_target:
        return False

    gam.challenge_progress += 1
    if gam.challenge_progress >= gam.challenge_target:
        gam.xp += CHALLENGE_BONUS_XP
        return True
    return False


def _check_achievements(gam: UserGamification) -> list[str]:
    unlocked = set(json.loads(gam.achievements or "[]"))
    newly = []
    for code, cfg in ACHIEVEMENTS.items():
        if code in unlocked:
            continue
        try:
            if cfg["check"](gam):
                unlocked.add(code)
                newly.append(code)
        except Exception:
            logger.exception("Ошибка проверки достижения %s", code)
    if newly:
        gam.achievements = json.dumps(sorted(unlocked), ensure_ascii=False)
    return newly


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


async def get_or_create_gam(user_id: int) -> UserGamification:
    async with SessionLocal() as session:
        gam = await session.get(UserGamification, user_id)
        if gam is None:
            gam = UserGamification(user_id=user_id)
            gam.xp = 0
            gam.streak = 0
            gam.challenge_progress = 0
            gam.challenge_target = 0
            gam.challenge_week = 0
            gam.achievements = "[]"
            gam.items_added = 0
            gam.outfits_created = 0
            gam.verdicts_done = 0
            gam.capsules_built = 0
            gam.laundry_done = 0
            session.add(gam)
        await _sync_existing_items(session, gam)
        await session.commit()
        await session.refresh(gam)
        return gam


async def _sync_existing_items(session, gam: UserGamification) -> None:
    """Синхронизировать счётчик добавленных вещей с фактической БД.

    Счётчики геймификации копятся с момента деплоя фичи. У пользователей,
    у которых вещи уже были в гардеробе на тот момент, items_added = 0,
    хотя реально вещей может быть много. Здесь мы доводим счётчик до
    фактического числа вещей в БД и начисляем XP за «пропущенные» вещи.
    """
    from app.db.repository import count_items

    try:
        actual = await count_items(gam.user_id)
    except Exception:
        logger.exception("Не удалось посчитать вещи для геймификации")
        return

    current = gam.items_added or 0
    if actual > current:
        diff = actual - current
        gam.items_added = actual
        gam.xp = (gam.xp or 0) + diff * EVENT_XP["item_added"]
        _check_achievements(gam)


async def record_event(
    user_id: int, event: str, today: str | None = None
) -> GamUpdate:
    """Записать событие и вернуть, что изменилось."""
    if event not in EVENT_XP:
        logger.warning("Неизвестное событие геймификации: %s", event)
        return GamUpdate()

    today = today or _today()
    async with SessionLocal() as session:
        gam = await session.get(UserGamification, user_id)
        if gam is None:
            gam = UserGamification(user_id=user_id)
            gam.xp = 0
            gam.streak = 0
            gam.challenge_progress = 0
            gam.challenge_target = 0
            gam.challenge_week = 0
            gam.achievements = "[]"
            gam.items_added = 0
            gam.outfits_created = 0
            gam.verdicts_done = 0
            gam.capsules_built = 0
            gam.laundry_done = 0
            session.add(gam)

        _update_streak(gam, today)

        xp_gain = EVENT_XP[event]
        before_level = level_from_xp(gam.xp or 0)
        gam.xp = (gam.xp or 0) + xp_gain
        after_level = level_from_xp(gam.xp)

        counter = EVENT_COUNTER[event]
        setattr(gam, counter, (getattr(gam, counter) or 0) + 1)

        challenge_done = _update_challenge(gam, event, today)
        new_ach = _check_achievements(gam)

        await session.commit()
        return GamUpdate(
            new_achievements=new_ach,
            leveled_up=after_level > before_level,
            new_level=after_level,
            challenge_completed=challenge_done,
            xp_gained=xp_gain,
        )
