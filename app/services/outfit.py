"""Подбор комплекта одежды по погоде и поводу."""

import logging
from dataclasses import dataclass, field

from app.db.models import WardrobeItem

logger = logging.getLogger(__name__)

MAX_ITEMS = 10
REQUIRED_CATEGORIES = ("верх", "низ", "обувь")
CATEGORY_ORDER = {"верх": 0, "низ": 1, "обувь": 2, "аксессуар": 3}


@dataclass
class OutfitResult:
    """Результат подбора комплекта.

    items — подобранные вещи (до MAX_ITEMS).
    strict_categories — категории, где вещь подошла по всем фильтрам.
    relaxed_categories — категории, где пришлось взять вещь не по погоде.
    missing_categories — категории, которых нет в гардеробе совсем.
    no_candidates — True, если вообще ничего не удалось подобрать
                    (гардероб пуст).
    """
    items: list[WardrobeItem] = field(default_factory=list)
    strict_categories: set[str] = field(default_factory=set)
    relaxed_categories: set[str] = field(default_factory=set)
    missing_categories: set[str] = field(default_factory=set)
    no_candidates: bool = False


def _required_warmth(temp: float) -> tuple[int, int]:
    """Диапазон допустимого warmth_level для температуры (жёсткий фильтр)."""
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


def _ideal_warmth(temp: float) -> int:
    """Идеальный warmth_level для температуры (мягкий приоритет)."""
    if temp >= 25:
        return 1
    if temp >= 18:
        return 2
    if temp >= 10:
        return 3
    if temp >= 0:
        return 4
    return 5


def _ideal_season(temp: float) -> str:
    """Идеальный сезон для температуры (мягкий приоритет)."""
    if temp >= 20:
        return "лето"
    if temp >= 10:
        return "демисезон"
    return "зима"


OCCASION_TO_FORMAL = {
    "работа": "business",
    "встреча": "business",
    "прогулка": "casual",
    "спорт": "sport",
    "другое": None,
}


def _score(
    item: WardrobeItem,
    formal: str | None,
    has_precip: bool,
    temp: float,
) -> int:
    """Рейтинг вещи. Чем больше — тем выше в списке выбора.

    Слагаемые:
      +10  formal_level совпал с поводом
      + 5  вещь waterproof, а сейчас дождь/снег
      +2 × max(0, 3 - |warmth_level - ideal_warmth|)  попадание в температуру
      + 3  season совпал с идеальным для температуры
    """
    s = 0

    if formal and item.formal_level == formal:
        s += 10

    if has_precip and item.waterproof:
        s += 5

    diff = abs(item.warmth_level - _ideal_warmth(temp))
    s += 2 * max(0, 3 - diff)

    if item.season == _ideal_season(temp):
        s += 3

    return s


def _accessory_matches(item: WardrobeItem, temp: float) -> bool:
    """Подходит ли аксессуар по температуре."""
    t = item.type.lower()
    hat_kw = ("шапк", "бини", "бейсболк", "шляп", "кепк")
    if temp < 0:
        return (
            any(k in t for k in hat_kw)
            or "шарф" in t
            or "перчат" in t
        )
    if temp < 5:
        return any(k in t for k in hat_kw)
    return False


def select_outfit(
    items: list[WardrobeItem],
    weather: dict,
    occasion: str,
) -> OutfitResult:
    """Подобрать комплект: не более MAX_ITEMS вещей.

    Всегда старается собрать хоть что-то, если в гардеробе есть вещи.
    Возвращает OutfitResult с информацией о слабых местах.

    items должны быть отсортированы по created_at DESC
    (это гарантирует get_items).
    """
    result = OutfitResult()

    temp = weather.get("temp", 15)
    has_precip = weather.get("has_precipitation", False)
    min_w, max_w = _required_warmth(temp)
    formal = OCCASION_TO_FORMAL.get(occasion)

    by_cat: dict[str, list[WardrobeItem]] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    def fits(i: WardrobeItem) -> bool:
        return min_w <= i.warmth_level <= max_w

    def rank(lst: list[WardrobeItem]) -> list[WardrobeItem]:
        return sorted(
            lst,
            key=lambda i: -_score(i, formal, has_precip, temp),
        )

    # Никаких вещей вообще
    if not items:
        result.no_candidates = True
        return result

    chosen: list[WardrobeItem] = []

    # 1. Обязательные категории: по одной вещи.
    #    Сначала strict (по warmth). Если не нашлось — relaxed (любая),
    #    и помечаем категорию как relaxed. Если категории нет совсем —
    #    пишем в missing_categories.
    for cat in REQUIRED_CATEGORIES:
        cat_items = by_cat.get(cat, [])
        if not cat_items:
            result.missing_categories.add(cat)
            continue

        strict = [i for i in cat_items if fits(i)]
        if strict:
            chosen.append(rank(strict)[0])
            result.strict_categories.add(cat)
        else:
            chosen.append(rank(cat_items)[0])
            result.relaxed_categories.add(cat)

    # 2. Аксессуары — только если холодно.
    if temp < 5:
        acc_items = [
            i for i in by_cat.get("аксессуар", [])
            if _accessory_matches(i, temp)
        ]
        if acc_items:
            chosen.extend(rank(acc_items)[:3])

    # 3. Второй слой категории "верх", если есть место.
    if len(chosen) < MAX_ITEMS:
        upper = [
            i for i in by_cat.get("верх", [])
            if fits(i) and i.id not in {x.id for x in chosen}
        ]
        for item in rank(upper):
            chosen.append(item)
            break

    # 4. Добираем оставшиеся подходящие (приоритет — новые).
    if len(chosen) < MAX_ITEMS:
        chosen_ids = {i.id for i in chosen}
        for item in items:
            if len(chosen) >= MAX_ITEMS:
                break
            if item.id in chosen_ids:
                continue
            if item.category == "аксессуар":
                continue
            if not fits(item):
                continue
            chosen.append(item)
            chosen_ids.add(item.id)

    # 5. Обрезка до MAX_ITEMS.
    if len(chosen) > MAX_ITEMS:
        chosen.sort(key=lambda i: i.created_at, reverse=True)
        chosen = chosen[:MAX_ITEMS]

    # Финальная сортировка по категориям.
    chosen.sort(key=lambda i: CATEGORY_ORDER.get(i.category, 9))

    logger.info(
        "Подбор: temp=%.1f has_precip=%s occasion=%s → %d вещей. "
        "strict=%s relaxed=%s missing=%s",
        temp, has_precip, occasion, len(chosen),
        result.strict_categories,
        result.relaxed_categories,
        result.missing_categories,
    )

    result.items = chosen
    return result