"""Подбор комплекта одежды: новый математический алгоритм + fallback."""

import logging
from dataclasses import dataclass, field

from app.db.models import WardrobeItem
from app.db.repository import get_colors, get_accessory_occasions

logger = logging.getLogger(__name__)


def _is_dress(item: WardrobeItem) -> bool:
    """Платье/сарафан/комбинезон — отдельная категория «верх», не требующая низа."""
    if not item:
        return False
    t = item.type.lower()
    return any(k in t for k in ["платье", "сарафан", "комбинезон"])


# ---------- Константы ----------

CATEGORY_ORDER = {"верх": 0, "низ": 1, "обувь": 2, "аксессуар": 3}

WARM_MATERIALS = {"шерсть", "пух", "флис", "кашемир", "кожа"}
LIGHT_MATERIALS = {"лён", "хлопок", "шёлк", "вискоза", "трикотаж"}

NEUTRAL_COLORS = {
    "чёрный", "белый", "серый", "тёмно-серый", "светло-серый",
    "бежевый", "кремовый", "тёмно-синий", "коричневый", "тёмно-коричневый",
}

# Приоритеты стилей по поводам
OCCASION_TO_STYLE_PRIORITY = {
    "спорт":    {"sport": 1.0, "casual": 0.6, "business": 0.1},
    "работа":   {"business": 1.0, "casual": 0.6, "sport": 0.1},
    "встреча":  {"business": 1.0, "casual": 0.6, "sport": 0.1},
    "прогулка": {"casual": 1.0, "sport": 0.8, "business": 0.4},
    "другое":   {"casual": 1.0, "sport": 0.8, "business": 0.5},
}

# Пороги стиля (п.8)
STYLE_HARD_CUT = 0.3      # ниже — не рассматриваем вообще
STYLE_FALLBACK_MAX = 0.5  # 0.3..0.5 — только если нет альтернативы

# Соседние сезоны для _season_fit (п.2)
SEASON_ADJACENT = {
    "лето": {"демисезон"},
    "демисезон": {"лето", "зима"},
    "зима": {"демисезон"},
}

MAX_COLOR_SCORE = 3.0
MAX_ACCENTS = 2  # максимум акцентных вещей в комплекте (п.6)


# ---------- Результат ----------

@dataclass
class OutfitResult:
    items: list[WardrobeItem] = field(default_factory=list)
    total_score: float = 0.0
    missing_categories: set[str] = field(default_factory=set)
    relaxed_categories: set[str] = field(default_factory=set)
    no_candidates: bool = False
    # Диагностика (п.11)
    score: float = 0.0
    reasons: dict = field(default_factory=dict)


# ===========================================================
# Чистые функции (п.13: тестируемость)
# ===========================================================

def _color_score(item: WardrobeItem) -> float:
    """Баллы за цвет: нейтральный 0.5, акцентный 1.0."""
    colors = get_colors(item)
    if not colors:
        return 0.5
    if any(c.lower() not in NEUTRAL_COLORS for c in colors):
        return 1.0
    return 0.5


def _get_style_priority(formal_level: str, occasion: str) -> float:
    """0.1..1.0 — насколько стиль подходит поводу."""
    priorities = OCCASION_TO_STYLE_PRIORITY.get(
        occasion, OCCASION_TO_STYLE_PRIORITY["прогулка"]
    )
    return priorities.get(formal_level, 0.5)


def _ideal_season(teff: float) -> str:
    if teff >= 20:
        return "лето"
    if teff >= 10:
        return "демисезон"
    return "зима"


def _get_targets(teff: float) -> dict:
    """Общие целевые warmth-диапазоны по эффективной температуре."""
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


def _get_layer_targets(teff: float) -> dict:
    """Отдельная цель warmth для base/mid/outer (п.2)."""
    if teff >= 24:
        return {"base": 1, "mid": 2, "outer": 3}
    if teff >= 18:
        return {"base": 2, "mid": 2, "outer": 3}
    if teff >= 14:
        return {"base": 2, "mid": 3, "outer": 4}
    if teff >= 10:
        return {"base": 3, "mid": 3, "outer": 4}
    if teff >= 6:
        return {"base": 3, "mid": 4, "outer": 5}
    if teff >= 2:
        return {"base": 3, "mid": 4, "outer": 5}
    if teff >= -3:
        return {"base": 4, "mid": 5, "outer": 5}
    if teff >= -10:
        return {"base": 4, "mid": 5, "outer": 5}
    return {"base": 5, "mid": 5, "outer": 5}


def _warmth_fit(item: WardrobeItem, w_target: float) -> float:
    if w_target <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(item.warmth_level - w_target) / 5.0)


def _waterproof_fit(item: WardrobeItem, need_waterproof: bool) -> float:
    """п.2 + п.4: в сухую погоду waterproof не штрафует."""
    if not need_waterproof:
        return 1.0
    return 1.0 if item.waterproof else 0.3


def _season_fit(item: WardrobeItem, teff: float) -> float:
    """п.2: учёт сезона."""
    ideal = _ideal_season(teff)
    if item.season == "универсальная" or item.season == ideal:
        return 1.0
    if item.season in SEASON_ADJACENT.get(ideal, set()):
        return 0.6
    return 0.2


def _material_fit(item: WardrobeItem, teff: float) -> float:
    mat = (item.material or "").lower()
    if not mat:
        return 0.5
    if teff > 22 and any(m in mat for m in LIGHT_MATERIALS):
        return 1.0
    if teff < 5 and any(m in mat for m in WARM_MATERIALS):
        return 1.0
    if teff < 5 and mat in ("лён", "шёлк"):
        return 0.0
    return 0.5


def _is_accent_color(color: str) -> bool:
    return color.lower() not in NEUTRAL_COLORS


def _color_harmony_penalty(combo: list[WardrobeItem]) -> tuple[float, int]:
    """п.6: не более MAX_ACCENTS акцентных вещей.

    Аксессуары в бюджет не считаются. Одна вещь = максимум один акцент.
    Возвращает (penalty, accent_count).
    """
    accent_count = 0
    for item in combo:
        if item.category == "аксессуар":
            continue
        for c in get_colors(item):
            if _is_accent_color(c):
                accent_count += 1
                break

    if accent_count <= MAX_ACCENTS - 1:
        return 0.0, accent_count
    if accent_count == MAX_ACCENTS:
        return -0.15, accent_count
    return -0.4 - (accent_count - MAX_ACCENTS) * 0.2, accent_count


def _layer_order_ok(combo: list[WardrobeItem]) -> bool:
    """п.12: base.warmth <= mid.warmth <= outer.warmth."""
    tops = [i for i in combo if i.category == "верх"]
    if len(tops) < 2:
        return True

    def layer_rank(i):
        return {"base": 0, "mid": 1, "outer": 2}.get(
            getattr(i, "layer", "base"), 0
        )

    tops_sorted = sorted(tops, key=layer_rank)
    warmth = [i.warmth_level for i in tops_sorted]
    return all(warmth[k] <= warmth[k + 1] for k in range(len(warmth) - 1))


def _accessory_matches(
    item: WardrobeItem, teff: float, wind: float, is_wet: bool
) -> bool:
    t = item.type.lower()

    hat_kw = ("шапк", "бини", "шляп", "кепк", "панам", "бейсбол")
    warm_acc_kw = ("шарф", "перчат", "варежк")
    rain_kw = ("зонт", "дождевик", "плащ")
    sun_kw = ("очки", "солнцезащит", "панам", "бейсбол", "кепк")

    # 1. Если идет дождь — зонт нужен при любой температуре
    if is_wet and any(k in t for k in rain_kw):
        return True

    # 2. Холодная погода
    if teff < 5:
        return any(k in t for k in hat_kw + warm_acc_kw)
    if teff < 10 and wind > 8:
        return any(k in t for k in warm_acc_kw)

    # 3. Жаркая погода
    if teff > 24 and any(k in t for k in sun_kw):
        return True

    return False


def _pick_accessories(
    items: list[WardrobeItem],
    teff: float,
    wind: float,
    is_wet: bool,
    occasion: str,
    current_color_score: float
) -> list[WardrobeItem]:
    """Подобрать до 2 аксессуаров под погоду, повод и цветовой бюджет."""
    accs = []
    for i in items:
        if i.category != "аксессуар":
            continue

        # 1. Проверка погоды (проходит ли аксессуар под текущую температуру/дождь)
        if not _accessory_matches(i, teff, wind, is_wet):
            continue

        # 2. Проверка повода.
        # ВАЖНОЕ ИСКЛЮЧЕНИЕ: Если это зонт и идет дождь, мы ИГНОРИРУЕМ список поводов.
        is_umbrella_in_rain = is_wet and "зонт" in i.type.lower()

        if not is_umbrella_in_rain and occasion != "другое":
            acc_occasions = get_accessory_occasions(i)
            if acc_occasions and occasion not in acc_occasions:
                continue

        accs.append(i)

    selected = []
    budget = current_color_score

    # 3. Приоритетное добавление зонта
    if is_wet:
        umbrellas = [a for a in accs if "зонт" in a.type.lower()]
        if umbrellas:
            best_umbrella = umbrellas[0]
            selected.append(best_umbrella)
            budget += _color_score(best_umbrella)
            accs.remove(best_umbrella) # Убираем, чтобы не добавить его еще раз ниже

    # 4. Добираем остальные аксессуары (до 2 штук)
    accs.sort(key=lambda a: _get_style_priority(a.formal_level, occasion), reverse=True)

    for acc in accs:
        if len(selected) >= 2:
            break # Максимум 2 аксессуара в образе

        c_score = _color_score(acc)
        if budget + c_score <= MAX_COLOR_SCORE:
            selected.append(acc)
            budget += c_score

    return selected


def _slot_of(item: WardrobeItem) -> str:
    if item.category == "верх":
        return "top"
    if item.category == "низ":
        return "bottom"
    return "shoes"


def _score_item(
    item: WardrobeItem,
    target: dict,
    layer_targets: dict,
    slot: str,
    teff: float,
    occasion: str,
    need_waterproof: bool,
) -> float:
    """Скор одной вещи с учётом style-приоритета."""
    priority = _get_style_priority(item.formal_level, occasion)
    if priority < STYLE_HARD_CUT:
        return -1000.0

    # ЖЕСТКИЙ ЗАПРЕТ: летние вещи при температуре ниже 12 градусов
    item_type = item.type.lower()
    if teff < 12 and any(k in item_type for k in ["шорт", "сандал", "майк", "топ", "панам"]):
        return -1000.0

    if slot == "top":
        layer = getattr(item, "layer", "base")
        w_target = layer_targets.get(layer, 3)
    elif slot == "bottom":
        w_target = (target["bottom_min"] + target["bottom_max"]) / 2
    else:
        w_target = (target["shoes_min"] + target["shoes_max"]) / 2

    warmth_fit = _warmth_fit(item, w_target)
    waterproof_fit = _waterproof_fit(item, need_waterproof)
    season_fit = _season_fit(item, teff)
    material_fit = _material_fit(item, teff)

    base = (
        0.45 * warmth_fit +
        0.25 * waterproof_fit +
        0.15 * season_fit +
        0.15 * material_fit
    )
    return base * priority


def _score_combo(
    combo: list[WardrobeItem],
    target: dict,
    layer_targets: dict,
    teff: float,
    occasion: str,
    need_waterproof: bool,
    missing: set[str],
) -> tuple[float, dict] | None:
    """п.1: скор всего комплекта. None → комбинацию отбрасываем."""
    reasons: dict = {}
    total = 0.0

    for item in combo:
        slot = _slot_of(item)
        item_score = _score_item(
            item, target, layer_targets, slot, teff, occasion, need_waterproof
        )
        if item_score <= -1000:
            return None
        total += item_score

        if _get_style_priority(item.formal_level, occasion) < STYLE_FALLBACK_MAX:
            reasons.setdefault("relaxed_style", []).append(item.category)

    # Цветовая гармония
    color_penalty, accent_count = _color_harmony_penalty(combo)
    total += color_penalty
    if accent_count > MAX_ACCENTS:
        reasons["too_many_accents"] = accent_count

    # Порядок слоёв (мягкий штраф, п.12)
    if not _layer_order_ok(combo):
        total -= 0.5
        reasons["layer_order_broken"] = True

    # п.3: need_mid / need_outer — мягкий штраф, если слоя нет в комбо
    has_mid = any(getattr(i, "layer", "") == "mid" for i in combo)
    has_outer = any(getattr(i, "layer", "") == "outer" for i in combo)
    if target["need_mid"] and not has_mid:
        total -= 0.2
        reasons["missing_mid"] = True
    if target["need_outer"] and not has_outer:
        total -= 0.2
        reasons["missing_outer"] = True

    # п.4: водостойкость — soft
    if need_waterproof:
        outer_or_shoes = [i for i in combo if i.category in ("верх", "обувь")]
        if not any(i.waterproof for i in outer_or_shoes):
            total -= 0.5
            reasons["no_waterproof"] = True

    # Пропущенные категории
    if missing:
        total -= 0.3 * len(missing)
        reasons["missing"] = list(missing)

    return total, reasons


def _pick_diverse(
    scored_combos: list[tuple[list, float, dict]],
    k: int,
    max_overlap: int = 2,
) -> list[tuple[list, float, dict]]:
    """п.7: top-K с ограничением на пересечение вещей."""
    if not scored_combos:
        return []

    selected: list[tuple[list, float, dict]] = []
    for combo, score, reasons in scored_combos:
        if len(selected) >= k:
            break
        combo_ids = {i.id for i in combo}
        overlap_ok = True
        for sel_combo, _, _ in selected:
            sel_ids = {i.id for i in sel_combo}
            if len(combo_ids & sel_ids) > max_overlap:
                overlap_ok = False
                break
        if overlap_ok:
            selected.append((combo, score, reasons))
    return selected


# ===========================================================
# Fallback (п.10: без formal)
# ===========================================================

def _required_warmth(teff: float) -> tuple[int, int]:
    if teff < -10: return 5, 5
    if teff < 0: return 4, 5
    if teff < 10: return 3, 5
    if teff < 18: return 2, 4
    if teff < 25: return 1, 3
    return 1, 2


def _ideal_warmth(teff: float) -> int:
    if teff >= 25: return 1
    if teff >= 18: return 2
    if teff >= 10: return 3
    if teff >= 0: return 4
    return 5


def _score_old(
    item: WardrobeItem, weather: dict, teff: float, occasion: str
) -> float:
    """Старый скоринг без formal."""
    has_precip = weather.get("has_precipitation", False)
    wind = weather.get("wind_speed", 0.0)

    priority = _get_style_priority(item.formal_level, occasion)
    if priority < STYLE_HARD_CUT:
        return -1000.0

    # ЖЕСТКИЙ ЗАПРЕТ: летние вещи при температуре ниже 12 градусов
    item_type = item.type.lower()
    if teff < 12 and any(k in item_type for k in ["шорт", "сандал", "майк", "топ", "панам"]):
        return -1000.0

    s = priority * 50

    if has_precip and item.waterproof:
        s += 10
    if wind > 5.0:
        if item.category == "верх":
            layer = getattr(item, "layer", "base")
            if layer == "outer":
                s += 5
            elif layer == "base":
                s -= 5
        if item.waterproof:
            s += 3

    mat = (item.material or "").lower()
    if mat:
        if teff > 22:
            if any(m in mat for m in LIGHT_MATERIALS):
                s += 5
            if any(m in mat for m in WARM_MATERIALS):
                s -= 5
        elif teff < 5:
            if any(m in mat for m in WARM_MATERIALS):
                s += 5
            if mat in ("лён", "шёлк"):
                s -= 5

    diff = abs(item.warmth_level - _ideal_warmth(teff))
    s += 2 * max(0, 3 - diff)
    if item.season == _ideal_season(teff):
        s += 3
    return s


def _select_outfit_fallback(
    items: list[WardrobeItem], weather: dict, occasion: str
) -> list[OutfitResult]:
    """Упрощённый алгоритм — всегда list[OutfitResult] (п.11)."""
    if not items:
        return [OutfitResult(no_candidates=True)]

    temp = weather.get("feels_like", weather.get("temp", 15))
    wind = weather.get("wind_speed", 0)
    teff = temp - 2 if wind >= 8 else temp
    min_w, max_w = _required_warmth(teff)

    by_cat: dict[str, list] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    def fits(i):
        return min_w <= i.warmth_level <= max_w

    def rank(lst):
        return sorted(
            lst, key=lambda i: -_score_old(i, weather, teff, occasion)
        )

    chosen: list[WardrobeItem] = []
    missing: set[str] = set()
    relaxed: set[str] = set()

    def pick(candidates, cat_name):
        if not candidates:
            return None
        ranked = rank(candidates)
        if _score_old(ranked[0], weather, teff, occasion) < 0:
            relaxed.add(cat_name)
        budget = 0.0
        for item in ranked:
            if budget + _color_score(item) <= MAX_COLOR_SCORE:
                return item
        return ranked[0]

    tops = by_cat.get("верх", [])
    chosen_top = None
    if tops:
        strict = [i for i in tops if fits(i)]
        chosen_top = pick(strict if strict else tops, "верх")
        if chosen_top:
            chosen.append(chosen_top)
    else:
        missing.add("верх")

    # Если выбранное платье - низ не нужен
    is_dress = _is_dress(chosen_top)

    for cat in ("низ", "обувь"):
        if cat == "низ" and is_dress:
            continue  # Пропускаем поиск низа

        cat_items = by_cat.get(cat, [])
        if not cat_items:
            missing.add(cat)
            continue
        strict = [i for i in cat_items if fits(i)]
        it = pick(strict if strict else cat_items, cat)
        if it:
            chosen.append(it)

    # Аксессуары — до 2 штук, с учётом текущего цветового бюджета
    current_color_score = sum(_color_score(i) for i in chosen)
    picked_accs = _pick_accessories(
        items, teff, wind, weather.get("has_precipitation", False),
        occasion, current_color_score=current_color_score,
    )
    chosen.extend(picked_accs)

    chosen.sort(key=lambda i: CATEGORY_ORDER.get(i.category, 9))

    if not chosen:
        return [OutfitResult(no_candidates=True)]

    return [OutfitResult(
        items=chosen,
        missing_categories=missing,
        relaxed_categories=relaxed,
        total_score=sum(
            _score_old(i, weather, teff, occasion) for i in chosen
        ),
    )]


# ===========================================================
# Главная функция
# ===========================================================

def select_outfit(
    items: list[WardrobeItem], weather: dict, occasion: str
) -> list[OutfitResult]:
    """Новый алгоритм с fallback (п.11: всегда list[OutfitResult])."""
    if not items:
        return [OutfitResult(no_candidates=True)]

    temp = weather.get("feels_like", weather.get("temp", 15))
    wind = weather.get("wind_speed", 0)
    is_wet = weather.get("has_precipitation", False)

    teff = temp - 2 if wind >= 8 else temp
    target = _get_targets(teff)
    layer_targets = _get_layer_targets(teff)
    need_waterproof = is_wet

    # Раскладываем по слотам
    bases, mids, outers, bottoms, shoes = [], [], [], [], []
    for i in items:
        if i.category == "верх":
            layer = getattr(i, "layer", "base")
            if layer == "base":
                bases.append(i)
            elif layer == "mid":
                mids.append(i)
            else:
                outers.append(i)
        elif i.category == "низ":
            bottoms.append(i)
        elif i.category == "обувь":
            shoes.append(i)

    def top_n(lst, slot, n=4):
        if not lst:
            return []
        scored = [
            (
                i,
                _score_item(
                    i, target, layer_targets, slot, teff, occasion, need_waterproof
                ),
            )
            for i in lst
        ]
        return [i for i, s in sorted(scored, key=lambda x: -x[1]) if s > 0][:n]

    bases = top_n(bases, "top")
    mids = top_n(mids, "top")
    outers = top_n(outers, "top")
    bottoms = top_n(bottoms, "bottom")
    shoes = top_n(shoes, "shoes")

    missing = set()
    if not bases:
        missing.add("верх")

    # Проверяем, есть ли в гардеробе хоть одно платье
    has_any_dress = any(_is_dress(i) for i in items)
    if not bottoms and not has_any_dress:
        missing.add("низ")

    if not shoes:
        missing.add("обувь")

    base_opts = bases if bases else [None]
    mid_opts = mids if mids else [None]
    outer_opts = outers if outers else [None]
    bottom_opts = bottoms if bottoms else [None]
    shoe_opts = shoes if shoes else [None]

    # Генерируем комбинации
    raw_combos: list[list[WardrobeItem]] = []
    for b in base_opts:
        is_b_dress = _is_dress(b)
        # Если это платье - низ не перебираем
        current_bottoms = [None] if is_b_dress else bottom_opts

        for m in mid_opts:
            for o in outer_opts:
                if o and not b:
                    continue
                if m and not b:
                    continue

                w_sum = sum(i.warmth_level for i in [b, m, o] if i)
                if w_sum < target["top_min"] or w_sum > target["top_max"] + 2:
                    continue

                for bot in current_bottoms:
                    for sh in shoe_opts:
                        combo = [x for x in [b, m, o, bot, sh] if x]
                        if not combo:
                            continue

                        if sum(_color_score(i) for i in combo) > MAX_COLOR_SCORE:
                            continue

                        raw_combos.append(combo)

    # Скор каждой комбинации
    scored: list[tuple[list, float, dict]] = []
    for combo in raw_combos:
        res = _score_combo(
            combo, target, layer_targets, teff, occasion, need_waterproof, missing
        )
        if res is None:
            continue
        score, reasons = res
        scored.append((combo, score, reasons))

    if not scored:
        logger.info("Новый алгоритм не собрал комбинаций → fallback на старый")
        return _select_outfit_fallback(items, weather, occasion)

    # п.1: сортируем по combo_score, sum(warmth) как tie-breaker
    scored.sort(
        key=lambda x: (
            x[1],
            sum(i.warmth_level for i in x[0] if i.category == "верх"),
        ),
        reverse=True,
    )

    # п.7: разнообразие
    picked = _pick_diverse(scored, k=5, max_overlap=2)
    if not picked:
        picked = scored[:5]

    results: list[OutfitResult] = []
    for combo, score, reasons in picked:
        combo_with_acc = list(combo)

        # п.5: аксессуары — до 2 штук под текущий комплект
        if not any(i.category == "аксессуар" for i in combo_with_acc):
            current_color_score = sum(
                _color_score(i) for i in combo_with_acc
            )
            picked_accs = _pick_accessories(
                items, teff, wind, is_wet, occasion,
                current_color_score=current_color_score,
            )
            combo_with_acc.extend(picked_accs)

        combo_with_acc.sort(key=lambda i: CATEGORY_ORDER.get(i.category, 9))

        relaxed_cats = set(reasons.get("relaxed_style", []))

        results.append(OutfitResult(
            items=combo_with_acc,
            total_score=score,
            score=score,
            missing_categories=missing,
            relaxed_categories=relaxed_cats,
            reasons=reasons,
        ))

    return results