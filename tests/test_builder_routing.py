"""Тесты роутинга графа (app/graph/builder.py)."""

import pytest

from app.graph.builder import _match_free_text_intent, route_condition


# ---------------------------------------------------------------------------
# _match_free_text_intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("я постирал джинсы", "laundry_text"),
        ("футболка в стирке", "laundry_text"),
        ("джинсы высохли", "laundry_text"),
        ("химчистка куртки", "laundry_text"),
        ("покажи гардероб", "list"),
        ("подбери образ", "outfit_start"),
        ("мой стиль", "dna_start"),
        ("приговор", "verdict_start"),
        ("помощь", "help"),
        ("отмена", "cancel"),
        ("привет", None),
    ],
)
def test_free_text_intent(text, expected):
    assert _match_free_text_intent(text) == expected


# ---------------------------------------------------------------------------
# route_condition: колбэки
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"input_text": "list:toggle_laundry"}, "list_toggle_laundry"),
        ({"input_text": "list:toggle_overlay"}, "list_toggle_overlay"),
        ({"input_text": "laundry:set:42:1"}, "laundry_callback"),
        ({"input_text": "laundry:set:7:0"}, "laundry_callback"),
        ({"input_text": "capsule:like"}, "capsule_like"),
        ({"input_text": "capsule:dislike"}, "capsule_dislike_menu"),
        ({"input_text": "capsule:remove:5"}, "capsule_remove"),
        ({"input_text": "capsule:back"}, "capsule_start"),
        ({"input_text": "menu:capsule"}, "capsule_start"),
        ({"input_text": "/capsule"}, "capsule_start"),
        ({"input_text": "list:menu"}, "list_menu"),
        ({"input_text": "list:edit"}, "list_edit"),
        ({"input_text": "list:edit:thickness"}, "list_edit_field"),
    ],
)
def test_route_condition_callbacks(state, expected):
    assert route_condition(state) == expected


def test_route_condition_free_text_uses_intent():
    assert route_condition({"input_text": "я постирал джинсы"}) == "laundry_text"
    assert route_condition({"input_text": "привет"}) != "laundry_text"
