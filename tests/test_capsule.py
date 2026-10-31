"""Тесты алгоритма капсульного гардероба (app/services/capsule.py)."""

from app.services.capsule import generate_capsule
from app.services.outfit import CATEGORY_ORDER


def _closet(make_item):
    """Полный гардероб: 10 верха, 3 низа, 2 обувь, 2 аксессуара."""
    items = []
    for i in range(10):
        items.append(make_item(id=i + 1, category="верх", type=f"футболка{i}", warmth_level=3))
    for i in range(3):
        items.append(make_item(id=20 + i, category="низ", type="джинсы", warmth_level=3))
    for i in range(2):
        items.append(make_item(id=30 + i, category="обувь", type="кроссовки", warmth_level=3))
    for i in range(2):
        items.append(make_item(id=40 + i, category="аксессуар", type="шапка", layer="none"))
    return items


def test_too_few_items_returns_empty(make_item):
    items = [make_item(id=1), make_item(id=2), make_item(id=3)]
    assert generate_capsule(items, []) == []


def test_filters_laundry_and_disliked(make_item):
    items = _closet(make_item)
    items[0].is_in_laundry = True
    disliked_id = items[1].id

    capsule = generate_capsule(items, [disliked_id])
    ids = [i.id for i in capsule]
    assert items[0].id not in ids
    assert disliked_id not in ids


def test_respects_category_limits(make_item):
    capsule = generate_capsule(_closet(make_item), [])

    # Не больше 15 вещей суммарно
    assert len(capsule) <= 15

    # Лимиты по категориям
    limits = {"верх": 5, "низ": 3, "обувь": 2, "аксессуар": 2}
    counts = {}
    for item in capsule:
        counts[item.category] = counts.get(item.category, 0) + 1
    for cat, limit in limits.items():
        assert counts.get(cat, 0) <= limit, f"{cat}: {counts.get(cat)} > {limit}"


def test_sorted_by_category_order(make_item):
    capsule = generate_capsule(_closet(make_item), [])
    order = [CATEGORY_ORDER.get(i.category, 9) for i in capsule]
    assert order == sorted(order)
