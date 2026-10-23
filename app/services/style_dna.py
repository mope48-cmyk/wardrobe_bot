"""Стилевой ДНК-тест: детерминированный анализ гардероба.

Никакого LLM здесь нет — только счётчики. LLM подключается позже
в llm_stylist.generate_dna_comment() как «обёртка» в характер Васильева.
"""

from collections import Counter
from dataclasses import dataclass, field

from app.db.models import WardrobeItem
from app.db.repository import get_colors
from app.services.outfit import NEUTRAL_COLORS, _is_dress

# Минимум вещей, при котором ДНК-профиль имеет смысл
MIN_ITEMS_FOR_DNA = 5

# Порог «перекоса в casual»: доля casual от всех вещей
CASUAL_SKEW_THRESHOLD = 0.8


@dataclass
class StyleProfile:
    total_items: int

    # Стиль
    style_distribution: dict[str, int] = field(default_factory=dict)
    dominant_style: str = "casual"

    # Цвета
    top_colors: list[tuple[str, int]] = field(default_factory=list)
    neutral_pct: float = 0.0  # доля вещей, у которых все цвета нейтральные
    accent_pct: float = 0.0

    # Материалы
    top_materials: list[tuple[str, int]] = field(default_factory=list)

    # Тепло
    avg_warmth: float = 3.0
    warmth_label: str = "золотая середина"

    # Сезон
    dominant_season: str = "универсальная"

    # Слои
    layer_distribution: dict[str, int] = field(default_factory=dict)

    # Прочее
    accessories_count: int = 0
    waterproof_count: int = 0

    # Пробелы (человекочитаемые фразы)
    gaps: list[str] = field(default_factory=list)

    has_enough_items: bool = True


def _warmth_label(avg: float) -> str:
    if avg < 2.5:
        return "южанин — предпочитаете лёгкость"
    if avg <= 3.5:
        return "золотая середина"
    return "северянин — предпочитаете тепло"


def _is_item_neutral(item: WardrobeItem) -> bool:
    """Вещь считается нейтральной, если ВСЕ её цвета нейтральные."""
    colors = get_colors(item)
    if not colors:
        return True  # без цвета — считаем нейтральной (не наказываем)
    return all(c.lower() in NEUTRAL_COLORS for c in colors)


def analyze_wardrobe(items: list[WardrobeItem]) -> StyleProfile:
    """Посчитать метрики гардероба. Всегда возвращает профиль
    (даже если вещей меньше MIN_ITEMS_FOR_DNA — флаг has_enough_items)."""
    total = len(items)
    profile = StyleProfile(total_items=total, has_enough_items=total >= MIN_ITEMS_FOR_DNA)

    if total == 0:
        return profile

    # ---- Стиль ----
    style_counter = Counter(i.formal_level for i in items)
    profile.style_distribution = dict(style_counter)
    profile.dominant_style = style_counter.most_common(1)[0][0]

    # ---- Цвета ----
    color_counter: Counter[str] = Counter()
    for item in items:
        for c in get_colors(item):
            color_counter[c.lower()] += 1
    profile.top_colors = color_counter.most_common(3)

    neutral_items = sum(1 for i in items if _is_item_neutral(i))
    profile.neutral_pct = neutral_items / total
    profile.accent_pct = 1.0 - profile.neutral_pct

    # ---- Материалы ----
    material_counter = Counter(
        i.material.strip().lower() for i in items if i.material and i.material.strip()
    )
    profile.top_materials = material_counter.most_common(3)

    # ---- Тепло ----
    profile.avg_warmth = sum(i.warmth_level for i in items) / total
    profile.warmth_label = _warmth_label(profile.avg_warmth)

    # ---- Сезон ----
    season_counter = Counter(i.season for i in items)
    profile.dominant_season = season_counter.most_common(1)[0][0]

    # ---- Слои ----
    layer_counter = Counter(getattr(i, "layer", "base") for i in items)
    profile.layer_distribution = dict(layer_counter)

    # ---- Прочее ----
    profile.accessories_count = sum(1 for i in items if i.category == "аксессуар")
    profile.waterproof_count = sum(1 for i in items if i.waterproof)

    # ---- Пробелы ----
    cats = {i.category for i in items}
    layers = {getattr(i, "layer", "base") for i in items}

    if "верх" not in cats:
        profile.gaps.append("нет верха")
    if "низ" not in cats and not any(_is_dress(i) for i in items):
        profile.gaps.append("нет низа")
    if "обувь" not in cats:
        profile.gaps.append("нет обуви")
    if "outer" not in layers:
        profile.gaps.append("нет верхнего слоя (куртки/пальто)")
    if profile.waterproof_count == 0:
        profile.gaps.append("нет ни одной водостойкой вещи")
    if profile.accessories_count == 0:
        profile.gaps.append("нет аксессуаров")

    if (
        total >= MIN_ITEMS_FOR_DNA
        and profile.style_distribution.get("casual", 0) / total >= CASUAL_SKEW_THRESHOLD
        and profile.style_distribution.get("business", 0) == 0
    ):
        profile.gaps.append("сильный перекос в casual — нет ни одной business-вещи")

    return profile


def profile_to_llm_brief(profile: StyleProfile) -> str:
    """Сериализовать профиль в компактный текст для промпта LLM."""
    lines = [
        f"Всего вещей: {profile.total_items}",
        f"Доминирующий стиль: {profile.dominant_style} "
        f"(casual: {profile.style_distribution.get('casual', 0)}, "
        f"business: {profile.style_distribution.get('business', 0)}, "
        f"sport: {profile.style_distribution.get('sport', 0)})",
        f"Доля нейтральных вещей: {profile.neutral_pct:.0%}, "
        f"акцентных: {profile.accent_pct:.0%}",
    ]

    if profile.top_colors:
        colors_str = ", ".join(f"{c} ({n})" for c, n in profile.top_colors)
        lines.append(f"Топ-цвета: {colors_str}")

    if profile.top_materials:
        mat_str = ", ".join(f"{m} ({n})" for m, n in profile.top_materials)
        lines.append(f"Топ-материалы: {mat_str}")

    lines.append(
        f"Средний уровень тепла: {profile.avg_warmth:.1f}/5 "
        f"({profile.warmth_label})"
    )
    lines.append(f"Доминирующий сезон: {profile.dominant_season}")

    layers_present = ", ".join(
        f"{k}: {v}" for k, v in profile.layer_distribution.items()
    )
    if layers_present:
        lines.append(f"Слои: {layers_present}")

    lines.append(
        f"Аксессуаров: {profile.accessories_count}, "
        f"водостойких вещей: {profile.waterproof_count}"
    )

    if profile.gaps:
        lines.append("Пробелы в гардеробе: " + "; ".join(profile.gaps))
    else:
        lines.append("Пробелов не обнаружено — гардероб полный.")

    return "\n".join(lines)
