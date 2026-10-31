"""Тесты математического алгоритма подбора (app/services/outfit.py)."""


from app.services.outfit import (
    _pick_accessories,
    _score_item,
    select_outfit,
)

TARGET = {"bottom_min": 1, "bottom_max": 5, "shoes_min": 1, "shoes_max": 5}
LAYERS = {"base": 3, "mid": 3, "outer": 3}


# ---------------------------------------------------------------------------
# _score_item: толщина + капюшон
# ---------------------------------------------------------------------------


def test_score_thickness_penalizes_thin_outer_in_frost(make_item):
    thin = make_item(layer="outer", thickness=1, warmth_level=2)
    thick = make_item(layer="outer", thickness=5, warmth_level=5)

    s_thin = _score_item(thin, TARGET, LAYERS, "top", -5.0, "прогулка", False)
    s_thick = _score_item(thick, TARGET, LAYERS, "top", -5.0, "прогулка", False)
    assert s_thick > s_thin


def test_score_thickness_penalizes_thick_outer_in_warmth(make_item):
    thick = make_item(layer="outer", thickness=5, warmth_level=5)
    thin = make_item(layer="outer", thickness=1, warmth_level=2)

    s_thick = _score_item(thick, TARGET, LAYERS, "top", 20.0, "прогулка", False)
    s_thin = _score_item(thin, TARGET, LAYERS, "top", 20.0, "прогулка", False)
    assert s_thin > s_thick


def test_score_hood_bonus_in_rain(make_item):
    hood = make_item(has_hood=True, waterproof=True)
    no_hood = make_item(has_hood=False, waterproof=True)

    s_hood = _score_item(hood, TARGET, LAYERS, "top", 8.0, "прогулка", True)
    s_no_hood = _score_item(no_hood, TARGET, LAYERS, "top", 8.0, "прогулка", True)
    assert s_hood > s_no_hood


# ---------------------------------------------------------------------------
# _pick_accessories: зонт
# ---------------------------------------------------------------------------


def test_umbrella_always_added_in_rain(make_item):
    umbrella = make_item(id=1, category="аксессуар", type="зонт", layer="none")
    picked = _pick_accessories(
        [umbrella], 8.0, 5.0, True, "прогулка", 0.0
    )
    assert len(picked) == 1
    assert "зонт" in picked[0].type.lower()


def test_no_umbrella_when_not_wet(make_item):
    umbrella = make_item(id=1, category="аксессуар", type="зонт", layer="none")
    picked = _pick_accessories(
        [umbrella], 8.0, 5.0, False, "прогулка", 0.0
    )
    assert all("зонт" not in i.type.lower() for i in picked)


# ---------------------------------------------------------------------------
# select_outfit: общее поведение
# ---------------------------------------------------------------------------


def test_empty_items_returns_no_candidates(make_item):
    result = select_outfit([], {"temp": 20, "feels_like": 20}, "прогулка")
    assert result and result[0].no_candidates is True


def test_all_laundry_filtered_out(make_item):
    dirty = [
        make_item(is_in_laundry=True),
        make_item(is_in_laundry=True, category="низ", type="джинсы"),
        make_item(is_in_laundry=True, category="обувь", type="кроссовки"),
    ]
    result = select_outfit(dirty, {"temp": 20, "feels_like": 20}, "прогулка")
    assert result[0].no_candidates is True


def test_summer_items_rejected_in_cold(make_item):
    items = [
        make_item(id=1, type="футболка", season="лето", warmth_level=2),
        make_item(id=2, category="низ", type="джинсы", season="универсальная", warmth_level=3),
        make_item(id=3, category="обувь", type="кроссовки", season="универсальная", warmth_level=2),
        make_item(id=4, category="низ", type="шорты", season="лето", warmth_level=1),
        make_item(id=5, category="обувь", type="сандалии", season="лето", warmth_level=1),
    ]
    weather = {"temp": 5, "feels_like": 5, "wind_speed": 0, "has_precipitation": False}
    result = select_outfit(items, weather, "прогулка")

    best = result[0]
    assert best.no_candidates is False
    # Летние вещи имеют жёсткий запрет при teff < 15 — их быть не должно
    assert all("шорт" not in i.type and "сандал" not in i.type for i in best.items)


def test_rain_adds_umbrella_into_result(make_item):
    items = [
        make_item(id=1, type="футболка", warmth_level=3),
        make_item(id=2, category="низ", type="джинсы", warmth_level=3),
        make_item(id=3, category="обувь", type="ботинки", warmth_level=3, waterproof=True),
        make_item(id=4, category="аксессуар", type="зонт", layer="none"),
    ]
    weather = {"temp": 15, "feels_like": 15, "wind_speed": 0, "has_precipitation": True}
    result = select_outfit(items, weather, "прогулка")
    best = result[0]
    assert best.no_candidates is False
    assert any("зонт" in i.type.lower() for i in best.items)
