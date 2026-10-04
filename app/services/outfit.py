"""Подбор комплекта одежды по погоде и поводу."""

import logging

from app.db.models import WardrobeItem

logger = logging.getLogger(__name__)


def _required_warmth(temp: float) -> tuple[int, int]:
    """Диапазон подходящего warmth_level для температуры."""
    if temp < -10:
        return 5, 5
    if temp < 0:
        return 4, 5
    if temp < 10:
        return 3, 5
    if temp < 18:
        return 2, 4
    if temp < 25:
        return 1, 3
    return 1, 2


# Повод → формальность. None = без фильтра.
OCCASION_TO_FORMAL = {
    "работа": "business",
    "встреча": "business",
    "прогулка": "casual",
    "спорт": "sport",
    "другое": None,
}


def _pick_best(
    candidates: list[WardrobeItem],
    formal: str | None,
    has_precip: bool,
) -> WardrobeItem | None:
    """Выбрать лучшую вещь: сначала по совпадению formal, потом waterproof."""
    if not candidates:
        return None

    def score(item: WardrobeItem) -> int:
        s = 0
        if formal and item.formal_level == formal:
            s += 10
        if has_precip and item.waterproof:
            s += 5
        return -s  # отрицательный, чтобы сортировка была по убыванию

    candidates.sort(key=score)
    return candidates[0]


def select_outfit(
    items: list[WardrobeItem],
    weather: dict,
    occasion: str,
) -> dict[str, WardrobeItem]:
    """Подобрать комплект.

    Возвращает словарь вида:
        {"верх": item, "низ": item, "обувь": item, "шапка": item, ...}
    Если по категории ничего нет — ключа в словаре не будет.
    """
    temp = weather.get("temp", 15)
    has_precip = weather.get("has_precipitation", False)
    min_w, max_w = _required_warmth(temp)
    formal = OCCASION_TO_FORMAL.get(occasion)

    result: dict[str, WardrobeItem] = {}

    # Основные категории: верх, низ, обувь
    for category in ("верх", "низ", "обувь"):
        # Строгий фильтр по warmth
        candidates = [
            i for i in items
            if i.category == category and min_w <= i.warmth_level <= max_w
        ]
        # Если пусто — ослабляем (берём любые вещи этой категории)
        if not candidates:
            candidates = [i for i in items if i.category == category]

        pick = _pick_best(candidates, formal, has_precip)
        if pick:
            result[category] = pick

    # Аксессуары для холода
    accessory_by_keyword = {
        "шапка": ("шапк", "кепк", "бейсболк", "шляп", "бини"),
        "шарф": ("шарф",),
        "перчатки": ("перчат",),
    }

    if temp < 5:
        for key, keywords in accessory_by_keyword.items():
            if key in ("шарф", "перчатки") and temp >= 0:
                continue  # шарф и перчатки только при минусе
            for item in items:
                if item.category != "аксессуар":
                    continue
                t = item.type.lower()
                if any(kw in t for kw in keywords):
                    result[key] = item
                    break

    logger.info(
        "Подбор: temp=%.1f has_precip=%s occasion=%s formal=%s → %s",
        temp, has_precip, occasion, formal,
        {k: v.type for k, v in result.items()},
    )
    return result
