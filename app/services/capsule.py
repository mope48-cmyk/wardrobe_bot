"""Алгоритм подбора капсульного гардероба."""

import logging
from collections import Counter

from app.db.models import WardrobeItem
from app.services.outfit import CATEGORY_ORDER, select_outfit

logger = logging.getLogger(__name__)


def generate_capsule(
    items: list[WardrobeItem], disliked_ids: list[int]
) -> list[WardrobeItem]:
    """Выбирает 10-15 самых универсальных вещей."""
    # 1. Фильтруем: убираем вещи в стирке и те, что в дизлайках
    available = [
        i
        for i in items
        if not getattr(i, "is_in_laundry", False) and i.id not in disliked_ids
    ]

    if len(available) < 5:
        return []  # Слишком мало вещей для капсулы

    # 2. Задаем несколько сценариев (погода + повод) для тестирования сочетаний
    scenarios = [
        # (temp, feels_like, desc, wind, precip, occasion)
        (25, 25, "жарко", 0, False, "прогулка"),
        (15, 14, "прохладно", 3, False, "работа"),
        (5, 2, "холодно", 5, False, "прогулка"),
        (10, 8, "дождь", 4, True, "работа"),
        (18, 18, "тепло", 2, False, "спорт"),
    ]

    used_ids = Counter()

    # 3. Прогоняем алгоритм подбора под каждый сценарий
    for temp, feels, desc, wind, precip, occ in scenarios:
        weather = {
            "temp": temp,
            "feels_like": feels,
            "description": desc,
            "wind_speed": wind,
            "has_precipitation": precip,
        }
        outfits = select_outfit(available, weather, occ)

        # Берем топ-2 лучших образа из каждого сценария
        for outfit in outfits[:2]:
            if not outfit.no_candidates:
                for item in outfit.items or []:
                    used_ids[item.id] += 1

    # 4. Сортируем вещи по частоте использования
    # Если вещь не попала ни в один образ, ее счетчик равен 0
    scored_items = []
    for item in available:
        score = used_ids.get(item.id, 0)
        # Небольшой бонус за нейтральность (чтобы капсула была базовой)
        if getattr(item, "formal_level", "") == "casual":
            score += 0.5
        scored_items.append((item, score))

    # Сортируем по убыванию счета
    scored_items.sort(key=lambda x: x[1], reverse=True)

    # 5. Собираем до 15 вещей, соблюдая баланс категорий
    capsule = []
    cat_counts = {"верх": 0, "низ": 0, "обувь": 0, "аксессуар": 0}

    # Лимиты (чтобы не набрать 15 футболок и 0 штанов)
    limits = {"верх": 5, "низ": 3, "обувь": 2, "аксессуар": 2}

    for item, score in scored_items:
        if len(capsule) >= 15:
            break
        if cat_counts[item.category] < limits.get(item.category, 3):
            capsule.append(item)
            cat_counts[item.category] += 1

    # Сортируем для красивого вывода (верх, низ, обувь...)
    capsule.sort(key=lambda i: CATEGORY_ORDER.get(i.category, 9))
    return capsule
