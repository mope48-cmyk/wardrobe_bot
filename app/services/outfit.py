"""Подбор комплекта одежды: новый математический алгоритм + fallback."""

import logging
from dataclasses import dataclass, field

from app.db.models import WardrobeItem
from app.db.repository import get_colors

logger = logging.getLogger(__name__)

# ---------- Константы ----------
CATEGORY_ORDER = {"верх": 0, "низ": 1, "обувь": 2, "аксессуар": 3}

WARM_MATERIALS = {"шерсть", "пух", "флис", "кашемир", "кожа"}
LIGHT_MATERIALS = {"лён", "хлопок", "шёлк", "вискоза", "трикотаж"}

NEUTRAL_COLORS = {
    "чёрный", "белый", "серый", "тёмно-серый", "светло-серый", 
    "бежевый", "кремовый", "тёмно-синий", "коричневый", "тёмно-коричневый"
}

OCCASION_TO_FORMAL = {
    "работа": "business",
    "встреча": "business",
    "прогулка": "casual",
    "спорт": "sport",
    "другое": None,
}


@dataclass
class OutfitResult:
    items: list[WardrobeItem] = field(default_factory=list)
    total_score: float = 0.0
    missing_categories: set[str] = field(default_factory=set)
    relaxed_categories: set[str] = field(default_factory=set)
    no_candidates: bool = False


# ---------- Цвета ----------

def _color_score(item: WardrobeItem) -> float:
    """Баллы за цвет: нейтральный 0.5, яркий 1.0."""
    colors = get_colors(item)
    if not colors:
        return 0.5
    if any(c.lower() not in NEUTRAL_COLORS for c in colors):
        return 1.0
    return 0.5


# ==========================================================
# СТАРЫЙ АЛГОРИТМ (FALLBACK)
# ==========================================================

def _required_warmth(feels_like: float) -> tuple[int, int]:
    if feels_like < -10: return 5, 5
    if feels_like < 0: return 4, 5
    if feels_like < 10: return 3, 5
    if feels_like < 18: return 2, 4
    if feels_like < 25: return 1, 3
    return 1, 2


def _ideal_warmth(temp: float) -> int:
    if temp >= 25: return 1
    if temp >= 18: return 2
    if temp >= 10: return 3
    if temp >= 0: return 4
    return 5


def _ideal_season(temp: float) -> str:
    if temp >= 20: return "лето"
    if temp >= 10: return "демисезон"
    return "зима"


def _score_old(item, formal, weather, occasion) -> int:
    temp = weather.get("feels_like", weather.get("temp", 15))
    has_precip = weather.get("has_precipitation", False)
    wind = weather.get("wind_speed", 0.0)
    s = 0

    if formal and item.formal_level == formal: s += 10
    if has_precip and item.waterproof: s += 10

    if wind > 5.0:
        if item.category == "верх":
            layer = getattr(item, "layer", "base")
            if layer == "outer": s += 5
            elif layer == "base": s -= 5
        if item.waterproof: s += 3
        if item.type in ("парка", "пуховик", "куртка"): s += 3

    mat = (item.material or "").lower()
    if mat:
        if temp > 22:
            if any(m in mat for m in LIGHT_MATERIALS): s += 5
            if any(m in mat for m in WARM_MATERIALS): s -= 5
        elif temp < 5:
            if any(m in mat for m in WARM_MATERIALS): s += 5
            if mat in ("лён", "шёлк"): s -= 5

    if occasion == "спорт":
        if any(m in mat for m in LIGHT_MATERIALS): s += 3
        if item.formal_level == "sport": s += 5

    diff = abs(item.warmth_level - _ideal_warmth(temp))
    s += 2 * max(0, 3 - diff)
    if item.season == _ideal_season(temp): s += 3
    return s


def _accessory_matches(item, temp: float) -> bool:
    t = item.type.lower()
    hat_kw = ("шапк", "бини", "бейсболк", "шляп", "кепк")
    if temp < 0:
        return any(k in t for k in hat_kw) or "шарф" in t or "перчат" in t
    if temp < 5:
        return any(k in t for k in hat_kw)
    return False


def _select_outfit_fallback(items, weather, occasion) -> OutfitResult:
    """Простой алгоритм на случай, если новый не собрал ничего."""
    result = OutfitResult()
    temp = weather.get("feels_like", weather.get("temp", 15))
    min_w, max_w = _required_warmth(temp)
    formal = OCCASION_TO_FORMAL.get(occasion)

    by_cat: dict[str, list] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    def fits(i):
        return min_w <= i.warmth_level <= max_w

    def rank(lst):
        return sorted(lst, key=lambda i: -_score_old(i, formal, weather, occasion))

    def pick(candidates):
        ranked = rank(candidates)
        # Приоритет — пройти по цветовому бюджету
        budget = 0.0
        for item in ranked:
            if budget + _color_score(item) <= 3.0:
                return item
        return ranked[0] if ranked else None

    chosen = []
    missing = set()

    # Верх: просто берём лучшую из всех (без строгой иерархии)
    tops = by_cat.get("верх", [])
    if tops:
        strict = [i for i in tops if fits(i)]
        it = pick(strict) if strict else pick(tops)
        if it: chosen.append(it)
    else:
        missing.add("верх")

    for cat in ("низ", "обувь"):
        cat_items = by_cat.get(cat, [])
        if not cat_items:
            missing.add(cat)
            continue
        strict = [i for i in cat_items if fits(i)]
        it = pick(strict) if strict else pick(cat_items)
        if it: chosen.append(it)

    if temp < 5:
        accs = [i for i in by_cat.get("аксессуар", []) if _accessory_matches(i, temp)]
        if accs:
            chosen.append(rank(accs)[0])

    chosen.sort(key=lambda i: (CATEGORY_ORDER.get(i.category, 9), 0))
    result.items = chosen
    result.missing_categories = missing
    if not chosen:
        result.no_candidates = True
    return result


# ==========================================================
# НОВЫЙ АЛГОРИТМ
# ==========================================================

def _get_targets(teff: float) -> dict:
    if teff >= 26:      top = (1, 2, False, False)
    elif teff >= 22:    top = (1, 3, False, False)
    elif teff >= 18:    top = (2, 4, True, False)
    elif teff >= 14:    top = (3, 5, True, False)
    elif teff >= 10:    top = (4, 6, True, False)
    elif teff >= 6:     top = (5, 8, True, True)
    elif teff >= 2:     top = (6, 9, True, True)
    elif teff >= -3:    top = (7, 10, True, True)
    elif teff >= -10:   top = (9, 13, True, True)
    else:               top = (11, 15, True, True)

    if teff >= 24:      bottom = (1, 2)
    elif teff >= 18:    bottom = (2, 3)
    elif teff >= 10:    bottom = (3, 4)
    elif teff >= 0:     bottom = (4, 5)
    else:               bottom = (4, 5)

    if teff >= 18:      shoes = (1, 2)
    elif teff >= 10:    shoes = (2, 3)
    elif teff >= 0:     shoes = (3, 4)
    else:               shoes = (4, 5)

    return {
        "top_min": top[0], "top_max": top[1],
        "need_mid": top[2], "need_outer": top[3],
        "bottom_min": bottom[0], "bottom_max": bottom[1],
        "shoes_min": shoes[0], "shoes_max": shoes[1],
    }


def _score_item(item, target: dict, slot: str, need_waterproof: bool) -> float:
    if slot == "top":
        layer = getattr(item, "layer", "base")
        if layer == "base": w_target = target["top_min"]
        elif layer == "mid": w_target = 3
        else: w_target = 4
    elif slot == "bottom":
        w_target = (target["bottom_min"] + target["bottom_max"]) / 2
    else:
        w_target = (target["shoes_min"] + target["shoes_max"]) / 2

    warmth_fit = max(0.0, 1.0 - abs(item.warmth_level - w_target) / 5.0)

    if need_waterproof:
        w_match = 1.0 if item.waterproof else 0.2
    else:
        w_match = 1.0 if not item.waterproof else 0.8

    s_match = 1.0 if item.season == "универсальная" else 0.8

    return 0.6 * warmth_fit + 0.3 * w_match + 0.1 * s_match


# ==========================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==========================================================

def select_outfit(items: list[WardrobeItem], weather: dict, occasion: str) -> list[OutfitResult]:
    """Возвращает список вариантов. Сначала пробует новый алгоритм,
    если не собрал — откатывается на старый."""
    if not items:
        return [OutfitResult(no_candidates=True)]

    temp = weather.get("feels_like", weather.get("temp", 15))
    wind = weather.get("wind_speed", 0)
    is_wet = weather.get("has_precipitation", False)

    teff = temp - 2 if wind >= 8 else temp
    target = _get_targets(teff)
    need_waterproof = is_wet

    # Раскладываем по слотам
    bases, mids, outers, bottoms, shoes = [], [], [], [], []
    for i in items:
        if i.category == "верх":
            layer = getattr(i, "layer", "base")
            if layer == "base": bases.append(i)
            elif layer == "mid": mids.append(i)
            else: outers.append(i)
        elif i.category == "низ": bottoms.append(i)
        elif i.category == "обувь": shoes.append(i)

    def top_n(lst, slot, n=4):
        if not lst:
            return []
        scored = [(i, _score_item(i, target, slot, need_waterproof)) for i in lst]
        return [i for i, s in sorted(scored, key=lambda x: -x[1]) if s > 0][:n]

    bases = top_n(bases, "top")
    mids = top_n(mids, "top")
    outers = top_n(outers, "top")
    bottoms = top_n(bottoms, "bottom")
    shoes = top_n(shoes, "shoes")

    missing = set()
    if not bases: missing.add("верх")
    if not bottoms: missing.add("низ")
    if not shoes: missing.add("обувь")

    base_opts = bases if bases else [None]
    mid_opts = mids if mids else [None]
    outer_opts = outers if outers else [None]
    bottom_opts = bottoms if bottoms else [None]
    shoe_opts = shoes if shoes else [None]

    combos = []
    for b in base_opts:
        for m in mid_opts:
            for o in outer_opts:
                # Строгая иерархия слоёв
                if o and not (b and m): continue
                if m and not b: continue

                w_sum = sum(i.warmth_level for i in [b, m, o] if i)
                if w_sum < target["top_min"] or w_sum > target["top_max"] + 1:
                    continue

                for bot in bottom_opts:
                    for sh in shoe_opts:
                        combo = [x for x in [b, m, o, bot, sh] if x]
                        if not combo: continue

                        # Цветовой бюджет
                        c_score = sum(_color_score(i) for i in combo)
                        if c_score > 3.0: continue

                        # Waterproof
                        if need_waterproof:
                            has_wp = any(i.waterproof for i in combo if i.category in ("верх", "обувь"))
                            if not has_wp: continue

                        combos.append(combo)

    # Если новый алгоритм ничего не собрал — fallback на старый
    if not combos:
        logger.info("Новый алгоритм не собрал комбинаций → fallback на старый")
        fallback = _select_outfit_fallback(items, weather, occasion)
        if fallback.items:
            return [fallback]
        return [OutfitResult(no_candidates=True)]

    # Сортируем: от самого тёплого к самому лёгкому
    combos.sort(
        key=lambda c: sum(i.warmth_level for i in c if i.category == "верх"),
        reverse=True,
    )

    results = []
    for c in combos[:5]:
        c.sort(key=lambda i: CATEGORY_ORDER.get(i.category, 9))
        results.append(OutfitResult(items=c, missing_categories=missing))

    return results