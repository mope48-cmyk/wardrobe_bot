"""Подбор комплекта одежды по погоде и поводу (математический алгоритм)."""

import logging
from dataclasses import dataclass, field

from app.db.models import WardrobeItem
from app.db.repository import get_colors

logger = logging.getLogger(__name__)

# --- Константы для цветов ---
NEUTRAL_COLORS = {
    "чёрный", "белый", "серый", "тёмно-серый", "светло-серый", 
    "бежевый", "кремовый", "тёмно-синий", "коричневый", "тёмно-коричневый"
}

@dataclass
class OutfitResult:
    items: list[WardrobeItem] = field(default_factory=list)
    total_score: float = 0.0
    missing_categories: set[str] = field(default_factory=set)
    relaxed_categories: set[str] = field(default_factory=set)
    no_candidates: bool = False

def _color_score(item: WardrobeItem) -> float:
    """Считает баллы за цвет вещи."""
    colors = get_colors(item)
    if not colors:
        return 0.5
    # Если есть хоть один не-нейтральный цвет — 1.0
    if any(c.lower() not in NEUTRAL_COLORS for c in colors):
        return 1.0
    return 0.5

def _get_targets(teff: float) -> dict:
    """Таблицы требуемого тепла по слотам."""
    # TOP_TABLE: (min_warmth, max_warmth, need_mid, need_outer)
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

def _score_item(item: WardrobeItem, target: dict, slot: str, need_waterproof: bool) -> float:
    """Оценка вещи от 0 до 1."""
    # 1. Warmth fit
    if slot == "top":
        if getattr(item, "layer", "base") == "base": w_target = target["top_min"]
        elif getattr(item, "layer", "base") == "mid": w_target = 3
        else: w_target = 4
    elif slot == "bottom":
        w_target = (target["bottom_min"] + target["bottom_max"]) / 2
    else:
        w_target = (target["shoes_min"] + target["shoes_max"]) / 2

    warmth_fit = max(0.0, 1.0 - abs(item.warmth_level - w_target) / 5.0)

    # 2. Waterproof
    if need_waterproof:
        w_match = 1.0 if item.waterproof else 0.2
    else:
        w_match = 1.0 if not item.waterproof else 0.8

    # 3. Season (грубая проверка)
    s_match = 1.0 if item.season == "универсальная" else 0.8

    # Итоговый score
    return 0.6 * warmth_fit + 0.3 * w_match + 0.1 * s_match

def select_outfit(items: list[WardrobeItem], weather: dict, occasion: str) -> list[OutfitResult]:
    """Возвращает список лучших комплектов (от самого точного к менее точным)."""
    if not items:
        return [OutfitResult(no_candidates=True)]

    # 1. Нормализация погоды
    temp = weather.get("feels_like", weather.get("temp", 15))
    wind = weather.get("wind_speed", 0)
    is_wet = weather.get("has_precipitation", False)
    
    teff = temp
    if wind >= 8: teff -= 2
    
    target = _get_targets(teff)
    need_waterproof = is_wet
    
    # 2. Раскладываем по слотам
    bases, mids, outers, bottoms, shoes = [], [], [], [], []
    for i in items:
        if i.category == "верх":
            if getattr(i, "layer", "base") == "base": bases.append(i)
            elif getattr(i, "layer", "base") == "mid": mids.append(i)
            else: outers.append(i)
        elif i.category == "низ": bottoms.append(i)
        elif i.category == "обувь": shoes.append(i)

    # 3. Скоринг и отсев
    def top_n(lst, slot, n=4):
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

    # 4. Генерация комбинаций
    combos = []
    
    base_opts = bases if bases else [None]
    mid_opts = mids if mids else [None]
    outer_opts = outers if outers else [None]
    bottom_opts = bottoms if bottoms else [None]
    shoe_opts = shoes if shoes else [None]

    for b in base_opts:
        for m in mid_opts:
            for o in outer_opts:
                # ЖЕСТКАЯ ИЕРАРХИЯ СЛОЕВ
                if o and not (b and m): continue
                if m and not b: continue

                # Проверка суммы тепла верха
                w_sum = sum(i.warmth_level for i in [b, m, o] if i)
                if w_sum < target["top_min"] or w_sum > target["top_max"] + 1:
                    continue

                for bot in bottom_opts:
                    for sh in shoe_opts:
                        combo = [x for x in [b, m, o, bot, sh] if x]
                        if not combo: continue
                        
                        # Проверка цветов (не больше 3 баллов)
                        c_score = sum(_color_score(i) for i in combo)
                        if c_score > 3.0: continue

                        # Проверка waterproof
                        if need_waterproof:
                            has_wp = any(i.waterproof for i in combo if i.category in ("верх", "обувь"))
                            if not has_wp: continue

                        combos.append(combo)

    # 5. Сортировка и упаковка
    # Сортируем по сумме warmth (для кнопок "теплее/легче")
    combos.sort(key=lambda c: sum(i.warmth_level for i in c if i.category=="верх"), reverse=True)
    
    results = []
    for c in combos[:5]: # Берем топ-5 для кнопок
        # Сортируем вещи внутри образа для красивого вывода
        c.sort(key=lambda i: {"верх": 0, "низ": 1, "обувь": 2}.get(i.category, 9))
        results.append(OutfitResult(items=c, missing_categories=missing))
        
    return results