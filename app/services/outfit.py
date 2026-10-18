"""Подбор комплекта одежды по погоде и поводу."""

import logging
from dataclasses import dataclass, field

from app.db.models import WardrobeItem
from app.db.repository import get_colors

logger = logging.getLogger(__name__)

MAX_ITEMS = 10
REQUIRED_CATEGORIES = ("верх", "низ", "обувь")
CATEGORY_ORDER = {"верх": 0, "низ": 1, "обувь": 2, "аксессуар": 3}

# Материалы, которые греют
WARM_MATERIALS = {"шерсть", "пух", "флис", "кашемир", "кожа"}
# Материалы, которые дышат
LIGHT_MATERIALS = {"лён", "хлопок", "шёлк", "вискоза", "трикотаж"}

# Нейтральные цвета — за них даём меньше баллов в цветовом бюджете
NEUTRAL_COLORS = {
    "чёрный", "тёмно-серый", "серый", "светло-серый",
    "белый", "кремовый",
}
MAX_COLOR_SCORE = 3.0


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


def _required_warmth(feels_like: float) -> tuple[int, int]:
    """Диапазон warmth_level по ощущаемой температуре."""
    if feels_like < -10:
        return 5, 5
    if feels_like < 0:
        return 4, 5
    if feels_like < 10:
        return 3, 5
    if feels_like < 18:
        return 2, 4
    if feels_like < 25:
        return 1, 3
    return 1, 2


def _ideal_warmth(temp: float) -> int:
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


def _color_score(item: WardrobeItem) -> float:
    """Считает баллы за цвет вещи.

    Нейтральные цвета (чёрный, серый, белый, бежевый/кремовый) — 0.5.
    Яркие цвета (красный, синий, зелёный, жёлтый и т.д.) — 1.0.
    Если цветов нет — считаем вещь нейтральной (0.5).
    """
    colors = get_colors(item)
    if not colors:
        return 0.5
    if any(c.lower() not in NEUTRAL_COLORS for c in colors):
        return 1.0
    return 0.5


def _score(
    item: WardrobeItem,
    formal: str | None,
    weather: dict,
    occasion: str,
) -> int:
    """Рейтинг вещи с учётом погоды, ветра, материалов и повода."""
    # Используем ощущаемую температуру
    temp = weather.get("feels_like", weather.get("temp", 15))
    has_precip = weather.get("has_precipitation", False)
    wind = weather.get("wind_speed", 0.0)

    s = 0

    # 1. Соответствие поводу
    if formal and item.formal_level == formal:
        s += 10

    # 2. Осадки (дождь/снег)
    if has_precip and item.waterproof:
        s += 10

    # 3. Ветер
    if wind > 5.0:
        if item.category == "верх":
            layer = getattr(item, "layer", "base")
            if layer == "outer":
                s += 5
            elif layer == "base":
                s -= 5
        if item.waterproof:
            s += 3
        if item.type in ("парка", "пуховик", "куртка"):
            s += 3

    # 4. Учёт материалов
    mat = (item.material or "").lower()
    if mat:
        if temp > 22:
            # В жару приоритет дышащим
            if any(m in mat for m in LIGHT_MATERIALS):
                s += 5
            if any(m in mat for m in WARM_MATERIALS):
                s -= 5
        elif temp < 5:
            # В холод приоритет греющим
            if any(m in mat for m in WARM_MATERIALS):
                s += 5
            if mat in ("лён", "шёлк"):
                s -= 5

    # 5. Спорт
    if occasion == "спорт":
        if any(m in mat for m in LIGHT_MATERIALS):
            s += 3
        if item.formal_level == "sport":
            s += 5

    # 6. Идеальная температура и сезон
    diff = abs(item.warmth_level - _ideal_warmth(temp))
    s += 2 * max(0, 3 - diff)
    if item.season == _ideal_season(temp):
        s += 3

    return s


def _accessory_matches(item: WardrobeItem, temp: float) -> bool:
    t = item.type.lower()
    hat_kw = ("шапк", "бини", "бейсболк", "шляп", "кепк")
    if temp < 0:
        return any(k in t for k in hat_kw) or "шарф" in t or "перчат" in t
    if temp < 5:
        return any(k in t for k in hat_kw)
    return False


def select_outfit(items: list[WardrobeItem], weather: dict, occasion: str):
    """Подобрать комплект с учётом слоёв и цветового бюджета."""
    result = OutfitResult()
    temp = weather.get("feels_like", weather.get("temp", 15))
    has_precip = weather.get("has_precipitation", False)
    min_w, max_w = _required_warmth(temp)
    formal = OCCASION_TO_FORMAL.get(occasion)

    by_cat: dict[str, list[WardrobeItem]] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    def fits(i: WardrobeItem) -> bool:
        return min_w <= i.warmth_level <= max_w

    def rank(lst: list[WardrobeItem]) -> list[WardrobeItem]:
        return sorted(lst, key=lambda i: -_score(i, formal, weather, occasion))

    if not items:
        result.no_candidates = True
        return result

    chosen: list[WardrobeItem] = []
    current_color_score = 0.0

    # Вспомогательная функция для выбора вещи с учётом цветового лимита
    def pick(candidates: list[WardrobeItem]) -> WardrobeItem:
        nonlocal current_color_score
        ranked = rank(candidates)
        for item in ranked:
            if item in chosen:
                continue
            if current_color_score + _color_score(item) <= MAX_COLOR_SCORE:
                current_color_score += _color_score(item)
                return item
        # Если лимит превышен, берём самый нейтральный вариант
        for item in sorted(candidates, key=lambda i: _color_score(i)):
            if item in chosen:
                continue
            current_color_score += _color_score(item)
            return item
        return ranked[0]

    # 1. Слои по температуре (P0.1)
    if temp < 0:
        required_layers = ["base", "mid", "outer"]
    elif temp <= 15:
        required_layers = ["base", "mid_or_outer"]  # 2 слоя
    else:
        required_layers = ["base"]  # 1 слой

    # 2. Верх по слоям
    for layer in required_layers:
        if layer == "mid_or_outer":
            layer_items = [
                i for i in by_cat.get("верх", [])
                if getattr(i, "layer", "base") in ("mid", "outer")
            ]
        else:
            layer_items = [
                i for i in by_cat.get("верх", [])
                if getattr(i, "layer", "base") == layer
            ]

        if not layer_items:
            if layer == "base":
                result.missing_categories.add("верх")
            continue

        strict = [i for i in layer_items if fits(i)]
        if strict:
            chosen.append(pick(strict))
            result.strict_categories.add("верх")
        else:
            chosen.append(pick(layer_items))
            result.relaxed_categories.add("верх")

    # 3. Низ и Обувь
    for cat in ("низ", "обувь"):
        cat_items = by_cat.get(cat, [])
        if not cat_items:
            result.missing_categories.add(cat)
            continue
        strict = [i for i in cat_items if fits(i)]
        if strict:
            chosen.append(pick(strict))
            result.strict_categories.add(cat)
        else:
            chosen.append(pick(cat_items))
            result.relaxed_categories.add(cat)

    # 4. Аксессуары
    if temp < 5:
        acc_items = [
            i for i in by_cat.get("аксессуар", [])
            if _accessory_matches(i, temp)
        ]
        if acc_items:
            for acc in rank(acc_items)[:3]:
                if current_color_score + _color_score(acc) <= MAX_COLOR_SCORE:
                    chosen.append(acc)
                    current_color_score += _color_score(acc)

    # 5. Сортировка
    def sort_key(item: WardrobeItem) -> tuple[int, int]:
        cat_order = CATEGORY_ORDER.get(item.category, 9)
        if item.category == "верх":
            layer_order = {"base": 0, "mid": 1, "outer": 2}.get(
                getattr(item, "layer", "base"), 9
            )
            return (cat_order, layer_order)
        return (cat_order, 0)

    chosen.sort(key=sort_key)
    result.items = chosen
    return result